#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(imputeLCMD))

option_list <- list(
    make_option("--input_path", type = "character"),
    make_option("--output_path", type = "character"),
    make_option("--method", type = "character"),
    make_option("--knn_neighbors", type = "integer")
)

opts <- parse_args(OptionParser(option_list = option_list))

table <- read.delim(
    opts$input_path,
    sep = "\t",
    row.names = 1,
    check.names = FALSE
)

values <- as.matrix(table)
storage.mode(values) <- "double"

imputed <- switch(
    opts$method,
    "qrilc" = impute.QRILC(values),
    "knn" = impute.wrapper.KNN(
        values,
        K = opts$knn_neighbors
    )
)

if (is.list(imputed) && !is.null(imputed$completeObs)) {
    imputed <- imputed$completeObs
} else if (is.list(imputed)) {
    imputed <- imputed[[1]]
}

imputed <- as.data.frame(imputed, check.names = FALSE)

write.table(
    imputed,
    file = opts$output_path,
    sep = "\t",
    quote = FALSE,
    col.names = NA
)
