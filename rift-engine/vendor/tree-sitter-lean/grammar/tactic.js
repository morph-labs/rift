module.exports = {
  tactics: $ => prec.left(
    seq('by', sep1_($._tactic, seq(optional(';'), $._newline))),
  ),

  apply_tactic: $ => seq('apply', $._expression),
  rewrite: $ => seq(choice('rewrite', 'rw'), $._expression),
  term: $ => seq('exact', $._expression),
  simp: $ => prec.right(seq(
    'simp',
    optional(field('extra', $.list)),
  )),
  trivial: $ => 'trivial',

  // src/Init/Tactics.lean
  intro: $ => prec.left(seq('intro', repeat($._expression))),
  rfl: $ => 'rfl',

  _user_tactic: $ => $._expression,

  _tactic: $ => choice(
    $.apply_tactic,
    $.rewrite,
    $.simp,
    $.term,
    $.trivial,

    $.intro,
    $.rfl,

    $._user_tactic,
  ),
}
