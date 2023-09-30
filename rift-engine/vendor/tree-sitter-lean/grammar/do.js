const {PREC} = require('./basic.js')
const {term} = require('./term.js')

// src/Lean/Parser/Do.lean
module.exports = {
  _left_arrow: $ => choice('<-', '←'),
  _do_seq_item: $ => prec.left(seq($._do_element, optional(';'))),
  _do_seq_indent: $ => prec.left(seq(repeat1($._do_seq_item))),
  _do_seq: $ => prec.left(choice($._do_seq_indent)),

  do_let: $ => prec(10, seq('let', optional('mut'), $._let_decl)),
  _do_id_decl: $ => prec(10, seq(
    $.identifier,
    optional($._type_spec),
    $._left_arrow,
    $._do_element,
  )),
  _do_pat_decl: $ => seq(
    $._expression,
    $._left_arrow,
    $._do_element,
    optional(seq('|', $._do_element)),
  ),
  do_let_arrow: $ => seq(
    'let',
    optional('mut'),
    choice($._do_id_decl, $._do_pat_decl),
  ),
  _else_if: $ => seq('else', 'if'),
  _do_if_let_pure: $ => seq(':=', $._expression),
  _do_if_let_bind: $ => seq('←', $._expression),
  _do_if_let: $ => seq(
      'let',
      $._expression,
      choice($._do_if_let_pure, $._do_if_let_bind),
  ),
  _do_if_prop: $ => prec(10, seq(optional($.identifier), $._expression)),
  _do_if_cond: $ => choice($._do_if_let, $._do_if_prop),
  do_if: $ => prec.left(PREC.lead, seq(
    'if',
    $._do_if_cond,
    'then',
    $._do_seq,
    repeat(seq($._else_if, $._do_if_cond, 'then', $._do_seq)),
  )),
  _do_for_decl: $ => seq($._expression, 'in', $._expression),
  do_for: $ => seq(
    'for',
    sep1($._do_for_decl, ','),
    'do',
    $._do_seq,
  ),
  do_unless: $ => seq('unless', $._expression, 'do', $._do_seq),
  do_catch: $ => seq(
    'catch',
    $._binder_ident,
    optional(seq(':', $._expression)),
    '=>',
    $._do_seq,
  ),
  do_finally: $ => seq('finally', $._do_seq),
  do_try: $ => seq(
    'try',
    $._do_seq,
    repeat(choice($.do_catch)),
    optional($.do_finally),
  ),
  do_return: $ => prec.left(PREC.lead,
    choice(
      seq('return', field('value', $._expression)),
    ),
  ),
  _do_expression: $ => $._expression,

  do: $ => prec(PREC.lead, seq('do', $._do_seq)),

  term_unless: $ => seq(
    'unless',
    term.forbid($, 'do'),
    'do',
    alias($._do_seq, $.do),
  ),

  _do_element: $ => choice(
    $.do_let,
    $.do_let_arrow,
    $.do_if,
    $.do_unless,
    $.do_for,
    $.do_return,
    $._do_expression,
  ),

  rules: {
    _left_arrow: $ => choice('<-', '←'),
    do_return: $ => prec.left(PREC.lead,
      seq('return', optional(field('value', $._expression))),
    ),
    _do_expression: $ => $._expression,
    _do_element: $ => choice(
      $._do_expression,
      $.assign,
      $.for_in,
      $.let_bind,
      $.let_mut,
      $.do_return,
    ),
  },
}
