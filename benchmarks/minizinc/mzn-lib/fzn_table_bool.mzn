%-----------------------------------------------------------------------------%
% A table constraint: table(x, t) represents the constraint x in t where we
% consider each row in t to be a tuple and t as a set of tuples.
%-----------------------------------------------------------------------------%
predicate turbo_fzn_table_bool(bool: positive, array[int] of var bool: x, array[int] of int: t);

predicate fzn_table_bool(array[int] of var bool: x, array[int, int] of bool: t) = turbo_fzn_table_bool(true,x,array1d(t));

