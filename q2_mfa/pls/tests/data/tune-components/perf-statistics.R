dim_names <- list(c("Overall.BER", "Overall.ER"), c("comp1", "comp2"))
list(
    WeightedVote.error.rate = list(
        max.dist = matrix(c(0.1, 0.3, 0.2, 0.4), nrow = 2, dimnames = dim_names),
        centroids.dist = matrix(c(0.5, 0.7, 0.6, 0.8), nrow = 2, dimnames = dim_names)
    ),
    WeightedVote.error.rate.sd = list(
        centroids.dist = matrix(
            c(0.05, 0.07, 0.06, 0.08), nrow = 2, dimnames = dim_names
        ),
        max.dist = matrix(
            c(0.04, 0.02, 0.03, 0.01), nrow = 2,
            dimnames = lapply(dim_names, rev)
        )
    ),
    choice.ncomp = list(
        WeightedVote = matrix(
            c(2L, 1L, 4L, 3L), nrow = 2,
            dimnames = list(c("Overall.ER", "Overall.BER"), c("max.dist", "centroids.dist"))
        )
    )
)
