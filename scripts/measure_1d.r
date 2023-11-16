library(DescTools)
library(data.table)
library(ordinal)
library(goftest)
library(ggplot2)

# AD avg, min, max - CVM avg, min, max - KS avg, min, max
stats <- c(0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 1, 0)

nor_offset = 1
exp_offset = 10
log_offset = 19
cau_offset = 28
wei_offset = 37
gum_offset = 46
sec_offset = 55

# test offsets
ad_offset = 0
cvm_offset = 3
ks_offset = 6

# enable anderson_darling, cramer_von_mises, ks tests
enable_tests = 0
# enable graph plots -- difference, errors, histograms
enable_plots = 0

max_sample_iter = 22

adv_mape_results <- list(c(), c(), c(), c(), c(), c(), c())
box_mape_results <- list(c(), c(), c(), c(), c(), c(), c())
pav_mape_results <- list(c(), c(), c(), c(), c(), c(), c())
distr_names <- list("Normal", "Exponential", "Logistic", "Cauchy", "Weibull", "Gumbel", "Hyperbolic Secant")

psech <- Vectorize(function(x,mu,sigma,log.p = FALSE) {
    logcdf <-  log(2) - log(pi) + log( atan( exp( 0.5*pi*(x-mu)/sigma ) ) )
    val <- ifelse(log.p, logcdf, exp(logcdf))
    return(val)
})

load_as_vector <- function(file_name, i, j = 1, suffix = "") {
    str = sprintf(file_name, i, j)
    str = paste(str, suffix, sep="")
    print(str)
    data_list <- fread(str)
    vector <- unlist(data_list)
    vector <- as.vector(vector, 'numeric')
    return(vector)
}

test.anderson_darling <- function(i, j, offset, distr_name, file_name, ...) {
    print(distr_name)
    vector <- load_as_vector(file_name, i, j)
    print(length(vector))
    result = AndersonDarlingTest(vector, ...)
    pval = result$p.value
    stats[offset] <<- stats[offset]+pval
    stats[offset + 1] <<- min(c(stats[offset + 1], pval))
    stats[offset + 2] <<- max(c(stats[offset + 2], pval))
    return(result)
}

test.cramer_von_mises <- function(i, j, offset, distr_name, file_name, ...) {
    print(distr_name)
    vector <- load_as_vector(file_name, i, j)
    print(length(vector))
    result = cvm.test(vector, ...)
    pval = result$p.value
    stats[offset + cvm_offset] <<- stats[offset]+pval
    stats[offset + cvm_offset + 1] <<- min(c(stats[offset + 1], pval))
    stats[offset + cvm_offset + 2] <<- max(c(stats[offset + 2], pval))
    return(result)
}

test.ks <- function(i, j, offset, distr_name, file_name, ...) {
    print(distr_name)
    vector <- load_as_vector(file_name, i, j)
    print(length(vector))
    result = ks.test(vector, ...)
    print(result)
    pval = result$p.value
    stats[offset + ks_offset] <<- stats[offset]+pval
    stats[offset + ks_offset + 1] <<- min(c(stats[offset + 1], pval))
    stats[offset + ks_offset + 2] <<- max(c(stats[offset + 2], pval))
    return(result)
}

test.plotDiff <- function(i, def_name, adv_name, box_name, pav_name, cdfFunctor, j)
{
    # default
    vector <- load_as_vector(def_name, i)
    # advanced
    vector2 <- load_as_vector(adv_name, i)
    # box
    vector3 <- load_as_vector(box_name, i)
    # pavel variant
    vector4 <- load_as_vector(pav_name, i)

    hist(vector, col ='coral2', main = distr_names[j], breaks = 50)
    hist(vector2, col = 'darkolivegreen3', breaks = 50)
    hist(vector3, col = 'deepskyblue3', breaks = 50)
    hist(vector4, col = 'red', breaks = 50)

    subtr <- function(x, baseline, treatment)  abs(ecdf(baseline)(x) - ecdf(treatment)(x))
    subtract_ref <- function(x, p1) abs(cdfFunctor(x) - ecdf(p1)(x))
    subtract_all <- function(x, p1, p2) abs(subtract_ref(x, p1) - subtract_ref(x, p2))

    a <- vector
    b <- vector2
    c <- vector3
    d <- vector4
    plot(ecdf(vector), col="green", lwd=1, main=paste(def_name, length(vector), sep=", "), xlab="x", ylab="f(x)", pch = ".")
    lines(ecdf(vector2), col="blue", pch = ".")
    lines(ecdf(vector3), col="purple", pch = ".")
    lines(ecdf(vector4), col="maroon1", pch = ".")
    curve(subtract_ref(x,a), from=min(a,b), to=max(a,b), col="red", add=TRUE)
    curve(subtract_ref(x,b), from=min(a,b), to=max(a,b), col="forestgreen", add=TRUE)
    curve(subtract_all(x,a,b), from=min(a,b), to=max(a,b), col="orange", add=TRUE)
    curve(subtract_all(x,a,c), from=min(a,c), to=max(a,c), col="dodgerblue1", add=TRUE)
    curve(subtract_all(x,a,d), from=min(a,d), to=max(a,d), col="maroon1", add=TRUE)
    legend( x= "topleft", y=0.92,
            legend=c("Default", "Advanced", "Box", "Pav", "|r - a|", "|r - b|", "||r - d| - |r - a||", "||r - d| - |r - b||", "||r - d| - |r - p||"),
            col=c("green", "blue", "purple", "maroon1", "red", "forestgreen", "orange", "dodgerblue1", "maroon1"),
            pch=c(19, 19, 19))
    curve(cdfFunctor, col="gray", add=TRUE)

    #curve(subtr(x,a,b), from=min(a,b), to=max(a,b), col="red", main="CDF: |(ref - default) - (ref - adv)|")
    #curve(subtract_ref(x,a), from=min(a,b), to=max(a,b), col="red", main="CDF: |ref - def|")
    #curve(subtract_ref(x,b), from=min(a,b), to=max(a,b), col="red", main="CDF: |ref - adv|")
    #curve(subtract_all(x,a,b), from=min(a,b), to=max(a,b), col="orange", main="CDF: ||ref - def| - |ref - adv||")
    #curve(subtract_all(x,a,c), from=min(a,c), to=max(a,c), col="dodgerblue1", main="CDF: ||ref - def| - |ref - box||")

    vector <- load_as_vector(def_name, 0, 0, "-chain")
    vector2 <- load_as_vector(adv_name, 0, 0, "-chain")
    vector3 <- load_as_vector(box_name, 0, 0, "-chain")
    vector4 <- load_as_vector(pav_name, 0, 0, "-chain")

    df <- data.frame(var = c(rep('Default', length(vector)), rep('Advanced', length(vector2)) ),
                       value = c(vector, vector2))

    ggResult <- ggplot(df, aes(x=value, fill=var)) +
            geom_histogram( color='#e9ecef', alpha=0.75, position='identity', bins=100) +
            scale_x_continuous(breaks = seq(0, 100, 5), lim = c(0, 100)) +
            scale_fill_manual(values = c("red", "green"))
    plot(ggResult)

    df <- data.frame(var = c(rep('Default', length(vector)), rep('Box', length(vector3)) ),
                       value = c(vector, vector3))

    ggResult <- ggplot(df, aes(x=value, fill=var)) +
            geom_histogram( color='#e9ecef', alpha=0.75, position='identity', bins=100) +
            scale_x_continuous(breaks = seq(0, 100, 5), lim = c(0, 100)) +
            scale_fill_manual(values = c("red", "dodgerblue1"))
    plot(ggResult)

    df <- data.frame(var = c(rep('Advanced', length(vector2)), rep('Box', length(vector3)) ),
                       value = c(vector2, vector3))

    ggResult <- ggplot(df, aes(x=value, fill=var)) +
            geom_histogram( color='#e9ecef', alpha=0.75, position='identity', bins=100) +
            scale_x_continuous(breaks = seq(0, 100, 5), lim = c(0, 100)) +
            scale_fill_manual(values = c("green", "dodgerblue1"))
    plot(ggResult)

    df <- data.frame(var = c(rep('Box', length(vector3)), rep('Pav', length(vector4)) ),
                           value = c(vector3, vector4))

    ggResult <- ggplot(df, aes(x=value, fill=var)) +
            geom_histogram( color='#e9ecef', alpha=0.75, position='identity', bins=100) +
            scale_x_continuous(breaks = seq(0, 100, 5), lim = c(0, 100)) +
            scale_fill_manual(values = c("dodgerblue1", "maroon1"))
    plot(ggResult)
}

# i - interval id
test.plotIntegralError <- function(i, distr_name, def_name, adv_name, box_name, pav_name, cdfFunctor, intArrIndex) {
    adv_mape_value <- 0
    box_mape_value <- 0
    pav_mape_value <- 0

    MAPE <- function(r, n) {
        return(100 * mean(abs(r - n) / r))
    }

    for (j in 1:10) {
        # default
        a <- load_as_vector(def_name, i, j)
        # advanced
        b <- load_as_vector(adv_name, i, j)
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
    }

    adv_mape_results[[intArrIndex]] <<- c(adv_mape_results[[intArrIndex]], adv_mape_value / 10)
    print(adv_mape_results[[intArrIndex]])
    box_mape_results[[intArrIndex]] <<- c(box_mape_results[[intArrIndex]], box_mape_value / 10)
    print(box_mape_results[[intArrIndex]])
    pav_mape_results[[intArrIndex]] <<- c(pav_mape_results[[intArrIndex]], pav_mape_value / 10)
    print(pav_mape_results[[intArrIndex]])
}

normFunc <- function(x) { return(pnorm(x, mean = 0, sd = 1)) }
expFunc <- function(x) { return(pexp(x, rate=1.5)) }
logFunc <- function(x) { return(plogis(x, location=2, scale=1)) }
cauFunc <- function(x) { return(pcauchy(x, location=0, scale=1)) }
weiFunc <- function(x) { return(pweibull(x, shape=5, scale=1)) }
gumFunc <- function(x) { return(pgumbel(x, location=1.5, scale=3)) }
secFunc <- function(x) { return(psech(x, 0, 1)) }

# integrate over sample intervals
for (i in 1:max_sample_iter) {
    print(i)

    test.plotIntegralError(
        i, "Normal",
        "../build/data/test-rsdef-normal-%d-%d.data",
        "../build/data/test-rsadv-normal-%d-%d.data",
        "../build/data/test-rsbox-normal-%d-%d.data",
        "../build/data/test-rspav-normal-%d-%d.data",
         normFunc, 1
    )

    test.plotIntegralError(
        i, "Exponential",
        "../build/data/test-rsdef-exponential-%d-%d.data",
        "../build/data/test-rsadv-exponential-%d-%d.data",
        "../build/data/test-rsbox-exponential-%d-%d.data",
        "../build/data/test-rspav-exponential-%d-%d.data",
        expFunc, 2
    )

    test.plotIntegralError(
        i, "Logistic",
        "../build/data/test-rsdef-logistic-%d-%d.data",
        "../build/data/test-rsadv-logistic-%d-%d.data",
        "../build/data/test-rsbox-logistic-%d-%d.data",
        "../build/data/test-rspav-logistic-%d-%d.data",
        logFunc, 3
    )

    test.plotIntegralError(
        i, "Cauchy",
        "../build/data/test-rsdef-cauchy-%d-%d.data",
        "../build/data/test-rsadv-cauchy-%d-%d.data",
        "../build/data/test-rsbox-cauchy-%d-%d.data",
        "../build/data/test-rspav-cauchy-%d-%d.data",
        cauFunc, 4
    )

    test.plotIntegralError(
        i, "Weibull",
        "../build/data/test-rsdef-weibull-%d-%d.data",
        "../build/data/test-rsadv-weibull-%d-%d.data",
        "../build/data/test-rsbox-weibull-%d-%d.data",
        "../build/data/test-rspav-weibull-%d-%d.data",
        weiFunc, 5
    )

    test.plotIntegralError(
        i, "Gumbel",
        "../build/data/test-rsdef-gumbel-%d-%d.data",
        "../build/data/test-rsadv-gumbel-%d-%d.data",
        "../build/data/test-rsbox-gumbel-%d-%d.data",
        "../build/data/test-rspav-gumbel-%d-%d.data",
        gumFunc, 6
    )

    test.plotIntegralError(
        i, "Hyperbolic Secant",
        "../build/data/test-rsdef-sech-%d-%d.data",
        "../build/data/test-rsadv-sech-%d-%d.data",
        "../build/data/test-rsbox-sech-%d-%d.data",
        "../build/data/test-rspav-sech-%d-%d.data",
        secFunc, 7
    )

    if (enable_plots) {
        test.plotDiff(i, "../build/data/test-rsdef-normal-%d-%d.data",
                         "../build/data/test-rsadv-normal-%d-%d.data",
                         "../build/data/test-rsbox-normal-%d-%d.data",
                         "../build/data/test-rspav-normal-%d-%d.data",
                         normFunc, 1)

        test.plotDiff(i, "../build/data/test-rsdef-exponential-%d-%d.data",
                         "../build/data/test-rsadv-exponential-%d-%d.data",
                         "../build/data/test-rsbox-exponential-%d-%d.data",
                         "../build/data/test-rspav-exponential-%d-%d.data",
                         expFunc, 2)

        test.plotDiff(i, "../build/data/test-rsdef-logistic-%d-%d.data",
                         "../build/data/test-rsadv-logistic-%d-%d.data",
                         "../build/data/test-rsbox-logistic-%d-%d.data",
                         "../build/data/test-rspav-logistic-%d-%d.data",
                         logFunc, 3)

        test.plotDiff(i, "../build/data/test-rsdef-cauchy-%d-%d.data",
                         "../build/data/test-rsadv-cauchy-%d-%d.data",
                         "../build/data/test-rsbox-cauchy-%d-%d.data",
                         "../build/data/test-rspav-cauchy-%d-%d.data",
                         cauFunc, 4)

        test.plotDiff(i, "../build/data/test-rsdef-weibull-%d-%d.data",
                         "../build/data/test-rsadv-weibull-%d-%d.data",
                         "../build/data/test-rsbox-weibull-%d-%d.data",
                         "../build/data/test-rspav-weibull-%d-%d.data",
                         weiFunc, 5)

        test.plotDiff(i, "../build/data/test-rsdef-gumbel-%d-%d.data",
                         "../build/data/test-rsadv-gumbel-%d-%d.data",
                         "../build/data/test-rsbox-gumbel-%d-%d.data",
                         "../build/data/test-rspav-gumbel-%d-%d.data",
                         gumFunc, 6)

        test.plotDiff(i, "../build/data/test-rsdef-sech-%d-%d.data",
                         "../build/data/test-rsadv-sech-%d-%d.data",
                         "../build/data/test-rsbox-sech-%d-%d.data",
                         "../build/data/test-rspav-sech-%d-%d.data",
                         secFunc, 7)
    }

    if (enable_tests) {
        test.anderson_darling(i, 1, nor_offset, "Normal", "../build/data/test-rsbox-normal-%d-%d.data", "pnorm", mean = 0, sd = 1)
        test.cramer_von_mises(i, 1, nor_offset, "Normal", "../build/data/test-rsbox-normal-%d-%d.data", "pnorm", mean = 0, sd = 1)
        test.ks(i, 1, nor_offset, "Normal", "../build/data/test-rsbox-normal-%d-%d.data", "pnorm", mean = 0, sd = 1)

        test.anderson_darling(i, 1, exp_offset, "Exponential", "../build/data/test-rsbox-exponential-%d-%d.data", "pexp", rate=1.5)
        test.cramer_von_mises(i, 1, exp_offset, "Exponential", "../build/data/test-rsbox-exponential-%d-%d.data", "pexp", rate=1.5)
        test.ks(i, 1, exp_offset, "Exponential", "../build/data/test-rsbox-exponential-%d-%d.data", "pexp", rate=1.5)

        test.anderson_darling(i, 1, log_offset, "Logistic", "../build/data/test-rsbox-logistic-%d-%d.data", "plogis", location=2, scale=1)
        test.cramer_von_mises(i, 1, log_offset, "Logistic", "../build/data/test-rsbox-logistic-%d-%d.data", "plogis", location=2, scale=1)
        test.ks(i, 1, log_offset, "Logistic", "../build/data/test-rsbox-logistic-%d-%d.data", "plogis", location=2, scale=1)

        test.anderson_darling(i, 1, cau_offset, "Cauchy", "../build/data/test-rsbox-cauchy-%d-%d.data", "pcauchy", location=0, scale=1)
        test.cramer_von_mises(i, 1, cau_offset, "Cauchy", "../build/data/test-rsbox-cauchy-%d-%d.data", "pcauchy", location=0, scale=1)
        test.ks(i, 1, cau_offset, "Cauchy", "../build/data/test-rsbox-cauchy-%d-%d.data", "pcauchy", location=0, scale=1)

        test.anderson_darling(i, 1, wei_offset, "Weibull", "../build/data/test-rsbox-weibull-%d-%d.data", "pweibull", shape=5, scale=1)
        test.cramer_von_mises(i, 1, wei_offset, "Weibull", "../build/data/test-rsbox-weibull-%d-%d.data", "pweibull", shape=5, scale=1)
        test.ks(i, 1, wei_offset, "Weibull", "../build/data/test-rsbox-weibull-%d-%d.data", "pweibull", shape=5, scale=1)

        test.anderson_darling(i, 1, gum_offset, "Gumbel", "../build/data/test-rsbox-gumbel-%d-%d.data", pgumbel, location=1.5, scale=3)
        test.cramer_von_mises(i, 1, gum_offset, "Gumbel", "../build/data/test-rsbox-gumbel-%d-%d.data", pgumbel, location=1.5, scale=3)
        test.ks(i, 1, gum_offset, "Gumbel", "../build/data/test-rsbox-gumbel-%d-%d.data", pgumbel, location=1.5, scale=3)

        test.anderson_darling(i, 1, sec_offset, "Hyperbolic Secant", "../build/data/test-rsbox-sech-%d-%d.data", psech, 0, 1)
        test.cramer_von_mises(i, 1, sec_offset, "Hyperbolic Secant", "../build/data/test-rsbox-sech-%d-%d.data", psech, 0, 1)
        test.ks(i, 1, sec_offset, "Hyperbolic Secant", "../build/data/test-rsbox-sech-%d-%d.data", psech, 0, 1)
    }
}

prnt.test <- function(x) {
   cat(x, sep="\n")
}

if (enable_tests) {
    print(stats)
    cat(c("\n"))
    prnt.test(c("AD - Normal", stats[nor_offset] / max_sample_iter, stats[nor_offset + 1], stats[nor_offset + 2]))
    prnt.test(c("CVM - Normal", stats[nor_offset + cvm_offset] / max_sample_iter, stats[nor_offset + cvm_offset + 1], stats[nor_offset + cvm_offset + 2]))
    prnt.test(c("KS - Normal", stats[nor_offset + ks_offset] / max_sample_iter, stats[nor_offset + ks_offset + 1], stats[nor_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Exponential", stats[exp_offset] / max_sample_iter, stats[exp_offset + 1], stats[exp_offset + 2]))
    prnt.test(c("CVM - Exponential", stats[exp_offset + cvm_offset] / max_sample_iter, stats[exp_offset + cvm_offset + 1], stats[exp_offset + cvm_offset + 2]))
    prnt.test(c("KS - Exponential", stats[exp_offset + ks_offset] / max_sample_iter, stats[exp_offset + ks_offset + 1], stats[exp_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Logistic", stats[log_offset] / max_sample_iter, stats[log_offset + 1], stats[log_offset + 2]))
    prnt.test(c("CVM - Logistic", stats[log_offset + cvm_offset] / max_sample_iter, stats[log_offset + cvm_offset + 1], stats[log_offset + cvm_offset + 2]))
    prnt.test(c("KS - Logistic", stats[log_offset + ks_offset] / max_sample_iter, stats[log_offset + ks_offset + 1], stats[log_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Cauchy", stats[cau_offset] / max_sample_iter, stats[cau_offset + 1], stats[cau_offset + 2]))
    prnt.test(c("CVM - Cauchy", stats[cau_offset + cvm_offset] / max_sample_iter, stats[cau_offset + cvm_offset + 1], stats[cau_offset + cvm_offset + 2]))
    prnt.test(c("KS - Cauchy", stats[cau_offset + ks_offset] / max_sample_iter, stats[cau_offset + ks_offset + 1], stats[cau_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Weibull", stats[wei_offset] / max_sample_iter, stats[wei_offset + 1], stats[wei_offset + 2]))
    prnt.test(c("CVM - Weibull", stats[wei_offset + cvm_offset] / max_sample_iter, stats[wei_offset + cvm_offset + 1], stats[wei_offset + cvm_offset + 2]))
    prnt.test(c("KS - Weibull", stats[wei_offset + ks_offset] / max_sample_iter, stats[wei_offset + ks_offset + 1], stats[wei_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Gumbel", stats[gum_offset] / max_sample_iter, stats[gum_offset + 1], stats[gum_offset + 2]))
    prnt.test(c("CVM - Gumbel", stats[gum_offset + cvm_offset] / max_sample_iter, stats[gum_offset + cvm_offset + 1], stats[gum_offset + cvm_offset + 2]))
    prnt.test(c("KS - Gumbel", stats[gum_offset + ks_offset] / max_sample_iter, stats[gum_offset + ks_offset + 1], stats[gum_offset + ks_offset + 2]))
    cat(c("\n"))
    prnt.test(c("AD - Hyperbolic Secant", stats[sec_offset] / max_sample_iter, stats[sec_offset + 1], stats[sec_offset + 2]))
    prnt.test(c("CVM - Hyperbolic Secant", stats[sec_offset + cvm_offset] / max_sample_iter, stats[sec_offset + cvm_offset + 1], stats[sec_offset + cvm_offset + 2]))
    prnt.test(c("KS - Hyperbolic Secant", stats[sec_offset + ks_offset] / max_sample_iter, stats[sec_offset + ks_offset + 1], stats[sec_offset + ks_offset + 2]))
}

cat("RS buffer MAPE:\n")
for (i in 1:7) {
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
for (i in 1:7) {
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
for (i in 1:7) {
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

for (i in 1:7) {
    plot(adv_mape_results[[i]], pch = 19, main = distr_names[[i]])
    lines(adv_mape_results[[i]], pch = ".", col = "red")
    lines(box_mape_results[[i]], pch = ".", col="cyan")
    lines(pav_mape_results[[i]], pch = ".", col="maroon1")
    axis(1, at = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
         labels = c(100, 500, 1000, 1500, 2000, 2500, 5000,
                    7500, 10000, 12500, 15000, 25000, 50000,
                    75000, 100000, 250000, 500000, 750000,
                    1000000, 1250000, 1500000, 2000000))
}