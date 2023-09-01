using CUDA.CUBLAS
using CUDA.CUDNN

import CUDA: libcudnn, @checked, @argout
import CUDA.CUDNN:
    @check, @runtime_ccall, @workspace, handle,
    cudnnStatus_t, cudnnConvolutionDescriptor_t,
    ConvDesc, TensorDesc, FilterDesc,
    CuArray, CuVector, CUDNNFloat, cdsize,
    cudnnConvolutionForward,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnConvolutionBwdDataAlgo_t,
    cudnnConvolutionBackwardFilter,
    cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnnConvolutionBwdFilterAlgo_t
import NNlib: depthwiseconv!, ∇depthwiseconv_filter!, ∇depthwiseconv_data!


# Including the deprecated macros: https://github.com/JuliaGPU/CUDA.jl/blob/6485df512ca838db66bbd3ded15e7e8836744912/lib/utils/call.jl

const CUDNNFloat = Union{Float16,Float32,Float64}

mutable struct ConvDesc
    ptr::cudnnConvolutionDescriptor_t
end

macro runtime_ccall(target, args...)
    # decode ccall function/library target
    Meta.isexpr(target, :tuple) || error("Expected (function_name, library) tuple")
    function_name, library = target.args

    # global const ref to hold the function pointer
    @gensym fptr_cache
    @eval __module__ begin
        # uses atomics (release store, acquire load) for thread safety.
        # see https://github.com/JuliaGPU/CUDAapi.jl/issues/106 for details
        const $fptr_cache = Threads.Atomic{UInt}(0)
    end

    return quote
        # use a closure to hold the lookup and avoid code bloat in the caller
        @noinline function cache_fptr!()
            library = Libdl.dlopen($(esc(library)))
            $(esc(fptr_cache))[] = Libdl.dlsym(library, $(esc(function_name)))

            $(esc(fptr_cache))[]
        end

        fptr = $(esc(fptr_cache))[]
        if fptr == 0        # folded into the null check performed by ccall
            fptr = cache_fptr!()
        end

        ccall(reinterpret(Ptr{Cvoid}, fptr), $(map(esc, args)...))
    end

    return
end

macro workspace(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    sz = nothing
    eltyp = :UInt8
    fallback = nothing
    for kwarg in kwargs
        key,val = kwarg.args
        if key == :size
            sz = val
        elseif key == :eltyp
            eltyp = val
        elseif key == :fallback
            fallback = val
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    if sz === nothing
        throw(ArgumentError("@workspace macro needs a size argument"))
    end

    # break down the closure to a let block to prevent JuliaLang/julia#15276
    Meta.isexpr(code, :(->)) || throw(ArgumentError("@workspace macro should be applied to a closure"))
    length(code.args) == 2 || throw(ArgumentError("@workspace closure should take exactly one argument"))
    code_arg = code.args[1]
    code = code.args[2]

    return quote
        sz = $(esc(sz))
        workspace = nothing
        try
          while workspace === nothing || length(workspace) < sz
              workspace = CuArray{$(esc(eltyp))}(undef, sz)
              sz = $(esc(sz))
          end
        catch ex
            $fallback === nothing && rethrow()
            isa(ex, OutOfGPUMemoryError) || rethrow()
            workspace = CuArray{UInt8}(undef, $fallback)
        end

        let $(esc(code_arg)) = workspace
            ret = $(esc(code))
            CUDA.unsafe_free!(workspace)
            ret
        end
    end
end

macro argout(ex)
    Meta.isexpr(ex, :call) || throw(ArgumentError("@argout macro should be applied to a function call"))

    block = quote end

    # look for output arguments (`out(...)`)
    output_vars = []
    args = ex.args[2:end]
    for (i,arg) in enumerate(args)
        if Meta.isexpr(arg, :call) && arg.args[1] == :out
            # allocate a variable
            @gensym output_val
            push!(block.args, :($output_val = $(ex.args[i+1].args[2]))) # strip `output(...)`
            push!(output_vars, output_val)

            # replace the argument
            ex.args[i+1] = output_val
        end
    end

    # generate a return
    push!(block.args, ex)
    if isempty(output_vars)
        push!(block.args, :(nothing))
    elseif length(output_vars) == 1
        push!(block.args, :($(output_vars[1])))
    else
        push!(block.args, :(tuple($(output_vars...))))
    end

    esc(block)
end

macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # generate a "safe" version that performs a check
    safe_body = quote
        @check $body
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, Symbol("unsafe_", sig.args[1]), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return esc(:($safe_def, $unsafe_def))
end

const CuOrVector = Union{CuVector, Vector}
const CuOrMatrix = Union{CuMatrix, Matrix}
const CuOrArray = Union{CuArray, Array}

# Implement conversion to dense, diagonal matrix.

diagonal(x::CuArray{T, 1}) where T<:Real = convert(CuArray, Diagonal(x))

# Implement GPU support for depthwise separable convolutions.
"""
@checked function cudnnGetConvolutionGroupCount(convDesc, count)
    @runtime_ccall(
        (:cudnnGetConvolutionGroupCount, libcudnn()),
        cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Ptr{Cint}),
        convDesc,
        count
    )
end

@checked function cudnnSetConvolutionGroupCount(convDesc, count)
    @runtime_ccall(
        (:cudnnSetConvolutionGroupCount, libcudnn()),
        cudnnStatus_t,
        (cudnnConvolutionDescriptor_t, Cint),
        convDesc,
        count
    )
end
"""
function ConvDesc(T, cdims::DepthwiseConvDims)
    cd = Ref{cudnnConvolutionDescriptor_t}()
    CUDNN.cudnnCreateConvolutionDescriptor(cd)
    N = NNlib.spatial_dims(cdims)
    CUDNN.cudnnSetConvolutionNdDescriptor(
        cd[],
        N,
        # Asymmetric padding is not supported.
        cdsize(NNlib.padding(cdims)[1:2:end], N),
        cdsize(NNlib.stride(cdims), N),
        cdsize(NNlib.dilation(cdims), N),
        NNlib.flipkernel(cdims),
        CUDNN.cudnnDataType(T)
    )
    # Set number of groups equal to number of channels to get a depthwise
    # separable convolution.
    cudnnSetConvolutionGroupCount(cd[], NNlib.channels_in(cdims))
    this = ConvDesc(cd[])
    CUDNN.finalizer(CUDNN.unsafe_free!, this)
    return this
end

function cudnnConvolutionForward(
    y::CuArray{T, N},
    x::CuArray{T, N},
    w::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle(),
            TensorDesc(x),
            FilterDesc(w),
            ConvDesc(T, cdims),
            TensorDesc(y),
            cudnnConvolutionFwdAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionForward(
            handle(),
            Ref(T(alpha)),
            TensorDesc(x),
            x,
            FilterDesc(w),
            w,
            ConvDesc(T,cdims),
            cudnnConvolutionFwdAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            TensorDesc(y),
            y
        )
    end
    return y
end

function cudnnConvolutionBackwardData(
    dx::CuArray{T, N},
    w::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle(),
            FilterDesc(w),
            TensorDesc(dy),
            ConvDesc(T, cdims),
            TensorDesc(dx),
            cudnnConvolutionBwdDataAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionBackwardData(
            handle(),
            Ref(T(alpha)),
            FilterDesc(w),
            w,
            TensorDesc(dy),
            dy,
            ConvDesc(T, cdims),
            cudnnConvolutionBwdDataAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            TensorDesc(dx),
            dx
        )
    end
    return dx
end

function cudnnConvolutionBackwardFilter(
    dw::CuArray{T, N},
    x::CuArray{T, N},
    dy::CuArray{T, N},
    cdims::DepthwiseConvDims;
    algo=0,
    alpha=1,
    beta=0
) where {T, N}
    @workspace size = @argout(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle(),
            TensorDesc(x),
            TensorDesc(dy),
            ConvDesc(T, cdims),
            FilterDesc(dw),
            cudnnConvolutionBwdFilterAlgo_t(algo),
            out(Ref{Csize_t}())
        )
    )[] workspace -> begin
        cudnnConvolutionBackwardFilter(
            handle(),
            Ref(T(alpha)),
            TensorDesc(x),
            x,
            TensorDesc(dy),
            dy,
            ConvDesc(T, cdims),
            cudnnConvolutionBwdFilterAlgo_t(algo),
            workspace,
            sizeof(workspace),
            Ref(T(beta)),
            FilterDesc(dw),
            dw
        )
    end
    return dw
end

function depthwiseconv!(
    y::CuArray{T},
    x::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionForward(
        y,
        x,
        w,
        cdims;
        alpha=alpha,
        algo=algo
    )
    return y
end

function ∇depthwiseconv_filter!(
    dw::CuArray{T},
    x::CuArray{T},
    dy::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionBackwardFilter(
        dw,
        x,
        dy,
        cdims;
        alpha=alpha,
        algo=algo
    )
    return dw
end

function ∇depthwiseconv_data!(
    dx::CuArray{T},
    dy::CuArray{T},
    w::CuArray{T},
    cdims::DepthwiseConvDims;
    alpha=1,
    algo=0
) where T<:CUDNNFloat
    cudnnConvolutionBackwardData(
        dx,
        w,
        dy,
        cdims;
        alpha=alpha,
        algo=algo
    )
end
