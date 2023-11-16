library(DescTools)
library(data.table)
library(ordinal)
library(goftest)
library(ggplot2)
library(circular)

max_sample_iter = 22
max_functions = 2

adv_mape_results <- list(c(), c())
box_mape_results <- list(c(), c())
pav_mape_results <- list(c(), c())
distr_names <- list("Von Mises", "Wrapped Exponential")

load_as_vector <- function(file_name, i, j = 1, suffix = "") {
    str = sprintf(file_name, i, j)
    str = paste(str, suffix, sep="")
    print(str)
    data_list <- fread(str)
    vector <- unlist(data_list)
    vector <- as.vector(vector, 'numeric')
    return(vector)
}

prnt.test <- function(x) {
   cat(x, sep="\n")
}

vonMisesFunc <- function(x) { return(pvonmises(x, mu=circular(pi), kappa=2)) }

pwrappedexp <- Vectorize(function(x,lambda) {
    val <- (1 - exp(-lambda * x)) / (1 - exp(-lambda * 2*pi))
    return(val)
})

wrappedExp <- function(x) { return(pwrappedexp(x, 1)) }

test.plotIntegralError <- function(i, distr_name, def_name, box_name, pav_name, cdfFunctor, intArrIndex) {
    adv_mape_value <- 0
    box_mape_value <- 0
    pav_mape_value <- 0

    MAPE <- function(r, n) {
        return(100 * mean(abs(r - n) / r))
    }

    for (j in 1:10) {
        # default
        a <- load_as_vector(def_name, i, j)
        # box
        c <- load_as_vector(box_name, i, j)
        # pavel variant
        d <- load_as_vector(pav_name, i, j)

        fun_ecdf <- ecdf(a)
        val_ecdf <- fun_ecdf(a)
        val_cdf <- cdfFunctor(a)
        mape = MAPE(val_cdf, val_ecdf)
        adv_mape_value <- adv_mape_value + mape

        fun_ecdf <- ecdf(c)
        val_ecdf <- fun_ecdf(c)
        val_cdf <- cdfFunctor(c)
        mape = MAPE(val_cdf, val_ecdf)
        box_mape_value <- box_mape_value + mape

        fun_ecdf <- ecdf(d)
        val_ecdf <- fun_ecdf(d)
        val_cdf <- cdfFunctor(d)
        mape = MAPE(val_cdf, val_ecdf)
        pav_mape_value <- pav_mape_value + mape

        plot(ecdf(a), col="green", lwd=1, main=paste(def_name, length(a), sep=", "), xlab="x", ylab="f(x)", pch = ".")
        lines(ecdf(c), col="blue", pch = ".")
        lines(ecdf(d), col="maroon1", pch = ".")
        curve(cdfFunctor, col="gray", add=TRUE)

        hist(a, col ='green', main = distr_names[j], breaks = 50)
        hist(c, col = 'blue', breaks = 50)
        hist(d, col = 'maroon1', breaks = 50)
    }

    adv_mape_results[[intArrIndex]] <<- c(adv_mape_results[[intArrIndex]], adv_mape_value / 10)
    print(adv_mape_results[[intArrIndex]])
    box_mape_results[[intArrIndex]] <<- c(box_mape_results[[intArrIndex]], box_mape_value / 10)
    print(box_mape_results[[intArrIndex]])
    pav_mape_results[[intArrIndex]] <<- c(pav_mape_results[[intArrIndex]], pav_mape_value / 10)
    print(pav_mape_results[[intArrIndex]])
}

for (i in 1:max_sample_iter) {
    print(i)

    test.plotIntegralError(
        i, "Von Mises",
        "../build/data/test-rsdef-vonmises-%d-%d.data",
        "../build/data/test-rsbox-vonmises-%d-%d.data",
        "../build/data/test-rspav-vonmises-%d-%d.data",
         vonMisesFunc, 1
    )

    test.plotIntegralError(
        i, "Wrapped Exponential",
        "../build/data/test-rsdef-wrappedexp-%d-%d.data",
        "../build/data/test-rsbox-wrappedexp-%d-%d.data",
        "../build/data/test-rspav-wrappedexp-%d-%d.data",
         wrappedExp, 2
    )
}

cat("RS buffer MAPE:\n")
for (i in 1:max_functions) {
    j <- 0
    cat(distr_names[[i]])
    cat("\n")
    vec <- adv_mape_results[[i]]
    for (val in vec) {
        cat(c("(", j, ",", val, ")"), sep="")
        j <- j + 1
    }
    cat("\n")
}
cat("Box buffer MAPE:\n")
for (i in 1:max_functions) {
    j <- 0
    cat(distr_names[[i]])
    cat("\n")
    vec <- box_mape_results[[i]]
    for (val in vec) {
        cat(c("(", j, ",", val, ")"), sep="")
        j <- j + 1
    }
    cat("\n")
}

cat("Pav buffer MAPE:\n")
for (i in 1:max_functions) {
    j <- 0
    cat(distr_names[[i]])
    cat("\n")
    vec <- pav_mape_results[[i]]
    for (val in vec) {
        cat(c("(", j, ",", val, ")"), sep="")
        j <- j + 1
    }
    cat("\n")
}