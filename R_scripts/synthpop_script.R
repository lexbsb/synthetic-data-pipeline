# rm(list = ls())

# cat("Starting\n")

# .libPaths("C:\\Users\\PinhoUlianA\\AppData\\Local\\Programs\\R\\R-4.4.2\\library")

# library("synthpop")

# args <- commandArgs()

# for (arg in args) {
#   if (grepl("train", arg)) {
#     train_loc <- arg
#   }
#   if (grepl("_synth", arg)) {
#     synth_loc <-arg
#   }
# }

# cat(train_loc, "\n")
# cat(synth_loc, "\n")

# df_fix <- read.csv(train_loc)
# na_counts <- apply(df_fix, 2, function(x) sum(is.na(x)))
# df_last_ord <- df_fix[,order(na_counts, decreasing = FALSE)]
# empty_last = names(df_last_ord)
# mysyn_emp_last = syn(df_fix, minnumlevels = 10, visit.sequence = empty_last, print.flag = TRUE)

# tryCatch({
#   write.syn(mysyn_emp_last, filename = synth_loc, filetype = "csv", save.complete = FALSE, extended.info = FALSE) 
# }, error = function(e){
#   print(e)
# })


rm(list = ls())
cat("Starting\n")

# Se realmente precisa fixar a lib:
#.libPaths("C:/Users/PinhoUlianA/AppData/Local/Programs/R/R-4.4.2/library")

suppressPackageStartupMessages(library(synthpop))

# --- argumentos seguros: <train_csv> <out_csv>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Use: Rscript script.R <train_csv> <out_csv>")

train_loc <- args[1]
synth_loc <- args[2]

cat(train_loc, "\n")
cat(synth_loc, "\n")

# normaliza caminhos e garante pasta/ extensão
train_loc <- normalizePath(train_loc, winslash = "/", mustWork = TRUE)
if (!grepl("\\.csv$", synth_loc, ignore.case = TRUE)) synth_loc <- paste0(synth_loc, ".csv")
out_dir <- dirname(synth_loc)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- lê dados
df_fix <- read.csv(train_loc, check.names = TRUE)

# # forces zip_code as fator, if present
# if ("zip_code" %in% names(df_fix)) {
#   df_fix$zip_code <- as.factor(df_fix$zip_code)
# }

#cat("Structure of the BD being synthetised: \n")
#str(df_fix)

# --- ordena variáveis por NAs e usa como visit.sequence
na_counts <- colSums(is.na(df_fix))
visit_seq <- names(df_fix)[order(na_counts, decreasing = FALSE)]

mysyn_emp_last <- syn(
  df_fix,
  minnumlevels = 10,
  visit.sequence = visit_seq,
  print.flag = TRUE
)

# --- salva como CSV (mais simples/robusto que write.syn)
tryCatch({
  utils::write.csv(mysyn_emp_last$syn, file = synth_loc, row.names = FALSE)
  cat("Arquivo salvo em:", synth_loc, "\n")
}, error = function(e) {
  message("Erro ao salvar CSV em: ", synth_loc)
  message(conditionMessage(e))
  q(status = 1)
})