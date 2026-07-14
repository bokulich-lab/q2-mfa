#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(missForest))

option_list <- list(
    make_option("--input_path", type = "character"),
    make_option("--output_path", type = "character"),
    make_option("--ntree", type = "integer"),
    make_option("--threads", type = "integer"),
    make_option("--random_state", type = "integer")
)

opts <- parse_args(OptionParser(option_list = option_list))

table <- read.delim(
    opts$input_path,
    sep = "\t",
    row.names = 1,
    check.names = FALSE,
    na.strings = "NA"
)

if (!is.null(opts$random_state)) {
    set.seed(opts$random_state)
}

imputed <- missForest(
    xmis = table,
    ntree = opts$ntree,
    parallelize = "no",
    num.threads = opts$threads,
    verbose = FALSE
)$ximp

write.table(
    imputed,
    file = opts$output_path,
    sep = "\t",
    quote = FALSE,
    col.names = NA
)
