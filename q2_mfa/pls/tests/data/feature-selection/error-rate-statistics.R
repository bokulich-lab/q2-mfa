list(
    error.rate = matrix(
        c(0.20, 0.35, 0.15, 0.28), nrow = 2,
        dimnames = list(c("1_2", "2_1"), c("comp1", "comp2"))
    ),
    error.rate.sd = matrix(
        c(0.04, 0.02, 0.03, 0.01), nrow = 2,
        dimnames = list(c("2_1", "1_2"), c("comp2", "comp1"))
    )
)
