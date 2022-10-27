export anp_ex2

function anp_ex2(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer,
    args::Any
)
    dim_x = 17
    dim_y = 9
    num_noise_channels, noise = build_categorical_noise(dim_y=dim_y)
    return Model(
        Parallel(
            Chain(
                InputsCoder(),
                DeterministicLikelihood()
            ),
            Chain(
                attention(
                    dim_x             =dim_x,
                    dim_y             =dim_y,
                    dim_embedding     =dim_embedding,
                    num_heads         =num_encoder_heads,
                    num_encoder_layers=num_encoder_layers
                ),
                DeterministicLikelihood()
            ),
            Chain(
                MLPCoder(
                    batched_mlp(
                        dim_in    =dim_x + dim_y,
                        dim_hidden=dim_embedding,
                        dim_out   =dim_embedding,
                        num_layers=num_encoder_layers
                    ),
                    batched_mlp(
                        dim_in    =dim_embedding,
                        dim_hidden=dim_embedding,
                        dim_out   =2dim_embedding,
                        num_layers=num_encoder_layers
                    )
                ),
                HeterogeneousGaussianLikelihood()
            )
        ),
        Chain(
            Materialise(),
            batched_mlp(
                dim_in    =dim_x + 2dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =num_noise_channels,
                num_layers=num_decoder_layers
            ),
            noise
        )
    )
end
