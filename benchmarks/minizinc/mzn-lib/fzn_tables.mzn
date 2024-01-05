/** A table over set variables. */
predicate turbo_fzn_table_set_of_int(array[int] of var set of int: x, array[int] of set of int: t);
predicate fzn_table_set_of_int(array[int] of var set of int: x, array[int, int] of set of int: t) = turbo_fzn_table_set_of_int(x,array1d(t));

/** A compressed table where each cell of the table is a set `{v1,v2,...}`. We interpret the empty set as the star `*` (all values allowed). */
predicate turbo_fzn_compressed_table_int(array[int] of var int: x, array[int] of set of int: t);
predicate fzn_compressed_table_int(array[int] of var int: x, array[int, int] of set of int: t) = turbo_fzn_compressed_table_int(x,array1d(t));
