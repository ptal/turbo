/** A table over int variables. Use 2147483647 (MAX_INT) for the star (*) symbol. */
predicate turbo_fzn_short_table_int(bool: positive, array[int] of var set of int: x, array[int] of set of int: t);
predicate fzn_short_table_int(bool: positive, array[int] of var set of int: x, array[int, int] of set of int: t) = turbo_fzn_short_table_int(positive, x, array1d(t));

/** A table over set variables. Use the set {2147483647} for the star (*) symbol. */
predicate turbo_fzn_short_table_set_of_int(bool: positive, array[int] of var set of int: x, array[int] of set of int: t);
predicate fzn_short_table_set_of_int(bool: positive, array[int] of var set of int: x, array[int, int] of set of int: t) = turbo_fzn_short_table_set_of_int(positive, x, array1d(t));

/** A compressed table where each cell of the table is a set `{v1,v2,...}`. We interpret the empty set as the star `*` (all values allowed). */
predicate turbo_fzn_compressed_table_int(bool: positive, array[int] of var int: x, array[int] of set of int: t);
predicate fzn_compressed_table_int(bool: positive, array[int] of var int: x, array[int, int] of set of int: t) = turbo_fzn_compressed_table_int(positive, x, array1d(t));

/** A basic table over int variables. For each line of the table `t`, we must have twice the number of columns (for the lower and upper bound). Use 2147483647 (MAX_INT) for the star (*) symbol. */
predicate turbo_fzn_basic_table_int(bool: positive, array[int] of var int: x, array[int] of int: t);
predicate fzn_basic_table_int(bool: positive, array[int] of var int: x, array[int, int] of int: t) = turbo_fzn_basic_table_int(positive, x, array1d(t));

/** A basic table over set variables. For each line of the table `t`, we must have twice the number of columns (for the lower and upper bound). Use the empty set {} for the star (*) symbol. */
predicate turbo_fzn_basic_table_set_of_int(bool: positive, array[int] of var set of int: x, array[int] of set of int: t);
predicate fzn_basic_table_set_of_int(bool: positive, array[int] of var set of int: x, array[int, int] of set of int: t) = turbo_fzn_basic_table_set_of_int(positive, x, array1d(t));
