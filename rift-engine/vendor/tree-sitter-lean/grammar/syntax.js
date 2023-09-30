const

macro_rhs = (parser) => seq('`(', repeat1(parser), ')');


// src/Lean/Parser/Syntax.lean
module.exports = {
  _precedence: $ => seq(
    token.immediate(':'),
    choice($.number, 'max', 'arg', 'lead', 'min', 'min1'),
  ),
  _syntax_cat: $ => $.identifier,
  _syntax_atom: $ => $.string,
  _syntax: $ => choice(
    $._syntax_cat,
    $._syntax_atom,
  ),

  mixfix: $ => seq(
    optional($._attr_kind),
    choice('prefix', 'infix', 'infixl', 'infixr', 'postfix'),
    $._precedence,
    $.string,
    '=>',
    $._term,
  ),
  notation: $ => seq(
    optional($._attr_kind),
    'notation',
    $._expression,
    '=>',
    $._expression,
  ),
  macro_rules: $ => seq(
    optional($._attr_kind),
    'macro_rules',
    $._match_alts,
  ),
  syntax: $ => seq(
    optional($._attr_kind),
    'syntax',
    repeat1($._syntax),
    ':',
    $.identifier,
  ),
  _macro_arg: $ => seq(
    optional(seq($.identifier, token.immediate(':'))),
    $._syntax,
  ),
  _macro_tail_tactic: $ => seq(
    ' : ', 'tactic', '=>', macro_rhs($._tactic),
  ),
  _macro_tail_command: $ => seq(
    ' : ', 'command', '=>', macro_rhs($._command),
  ),
  // Too lazy to implement macro_tail_default with a stack for now...
  _macro_tail_default: $ => seq(
    ' : ', $.identifier, '=>', macro_rhs(choice($._tactic, $._command)),
  ),
  _macro_tail: $ => choice(
    $._macro_tail_tactic,
    $._macro_tail_command,
  ),
  macro: $ => seq(
    optional($._attr_kind),
    'macro',
    optional($._precedence),
    repeat1($._macro_arg),
    $._macro_tail,
  ),
  _elab_arg: $ => $._macro_arg,
  _elab_tail: $ => seq(
    ':',
    $.identifier,
    optional(seq('<=', $.identifier)),
    '=>',
    $._expression,
  ),
  elab: $ => seq(
    optional($._attr_kind),
    'elab',
    optional($._precedence),
    repeat1($._elab_arg),
    $._elab_tail,
  ),
}
