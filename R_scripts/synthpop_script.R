rm(list = ls())

print('Starting')

.libPaths('<path to local R lib>')

library("synthpop")

args <- commandArgs()

for (arg in args) {
  if (grepl("train", arg)) {
    train_loc <- arg
  }
  if (grepl("_synth", arg)) {
    synth_loc <-arg
  }
}

print(train_loc)
print(synth_loc)

df_fix <- read.csv(train_loc)
na_counts <- apply(df_fix, 2, function(x) sum(is.na(x)))
df_last_ord <- df_fix[,order(na_counts, decreasing = FALSE)]
empty_last = names(df_last_ord)
mysyn_emp_last = syn(df_fix, minnumlevels = 10, visit.sequence = empty_last)

tryCatch({
  write.syn(mysyn_emp_last, filename = synth_loc, filetype = "csv", save.complete = FALSE, extended.info = FALSE) 
}, error = function(e){
  print(e)
})



