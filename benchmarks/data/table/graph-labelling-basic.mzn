include "globals.mzn";
include "fzn_tables.mzn";

int: n;
array[1..n] of var 1..n: labels;

constraint labels[16] = 16;

int: rows;
array[1..rows, 1..2] of set of int: supports;

int: t;
array[1..t, 1..2] of int: indexes;

constraint forall(i in 1..t)
 (fzn_compressed_table_int([labels[indexes[i,1]+1], labels[indexes[i,2]+1]],
        supports));

constraint alldifferent(labels);

solve satisfy;