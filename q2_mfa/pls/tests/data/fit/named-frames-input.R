list(
    block_b = matrix(
        c(0.2, 0.4, 0.6, 0.8), nrow = 2, byrow = TRUE,
        dimnames = list(c("feature_b", "feature_a"), c("comp2", "comp1"))
    ),
    block_a = matrix(
        c(0.3, 0.5), nrow = 1,
        dimnames = list("feature_c", c("comp2", "comp1"))
    )
)
