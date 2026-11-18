list(
    comp1 = matrix(
        c(0.8, 0.01, 0.7, 0.02), nrow = 2, byrow = TRUE,
        dimnames = list(c("A vs rest", "B vs rest"),
                        c("AUC", "p-value"))
    ),
    comp2 = matrix(
        c(0.9, 0.03, 0.85, 0.04), nrow = 2, byrow = TRUE,
        dimnames = list(c("A vs rest", "B vs rest"),
                        c("AUC", "p-value"))
    )
)
