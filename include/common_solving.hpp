// Copyright 2023 Pierre Talbot

#ifndef COMMON_SOLVING_HPP
#define COMMON_SOLVING_HPP

#include <algorithm>
#include <chrono>
#include <thread>

#include "config.hpp"
#include "statistics.hpp"

#include "allocator.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

#include "vstore.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "pc.hpp"
#include "terms.hpp"
#include "fixpoint.hpp"
#include "search_tree.hpp"
#include "bab.hpp"

#include "value_order.hpp"
#include "variable_order.hpp"
#include "split.hpp"

#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

const AType sty = 0;
const AType pty = 1;
const AType tty = 2;
const AType split_ty = 3;
const AType bab_ty = 4;

#endif
