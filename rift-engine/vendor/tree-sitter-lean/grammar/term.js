const {PREC} = require('./basic.js')
const {Parser, min1} = require('./util.js')

const match_alt = ($, rhs_parser) => seq(
  '|',
  field('lhs', sep1($._expression, ',')),
  '=>',
  (typeof rhs_parser !== 'undefined') ? rhs_parser : $._expression,
)
const match_alts = ($, rhs_parser) => repeat1(match_alt($, rhs_parser))

const term = new Parser($ => [
  $.identifier,
  $.number,
  $.float,
  $.string,
  $.char,
  $.hole,
  $.synthetic_hole,
  $.sorry,
  $.cdot,
  $.parenthesized,
  $.anonymous_constructor,
  $.structure_instance,
  $.explicit,
  $.forall,
  $.true,
  $.false,
  $.match,
  $.borrowed,
  $.quoted_name,
  $.double_quoted_name,
  $.have,
  $.proj,
  $.arrow,
  $._do_term,
  $._command_term,
  $._notation_term,
  $._notation_extra_term,
  $.array,
  $.subarray,
  $.range,
  $.interpolated_string,
])

// src/Lean/Parser/Term.lean
module.exports = {
  term,
  match_alt,
  rules: {
    // FIXME: see name.cpp for the real definition...
    identifier: $ => choice(
      sep1($._identifier, token.immediate('.')),
      $._identifier,
      $._escaped_identifier,
    ),
    number: $ => /\d+/,
    float: $ => /\d+\.\d*/,
    string: $ => seq(
      '"',
      repeat(choice($.quoted_char, token.immediate(prec(1, /[^"\n\\]+/)),)),
      '"',
    ),
    char: $ => seq("'", choice($.quoted_char, /[^']/), "'"),
    hole: $ => "_",
    synthetic_hole: $ => seq("?", choice($.identifier, $.hole)),
    sorry: $ => 'sorry',
    cdot: $ => choice('.', '·'),
    type_ascription: $ => seq(':', field('type', $._expression)),
    tuple_tail: $ => seq(',', sep1($._expression, ',')),
    parenthesized: $ => seq(
      '(',
      optional(
        seq($._expression, optional(choice($.tuple_tail, $.type_ascription)))
      ),
      ')',
    ),
    anonymous_constructor: $ => seq('⟨', sep0($._expression, ','), '⟩'),
    _structure_instance_field: $ => seq(
      field('name', $.identifier),
      ':=',
      field('value', $._expression),
    ),
    structure_instance: $ => seq(
      '{',
      optional(field('extends', seq($._expression, 'with'))),
      // Unlike everywhere else, records seem OK with trailing commas.
      optional(sep1_($._structure_instance_field, ',')),
      field('type', optional($._type_spec)),
      '}',
    ),
    _type_spec: $ => field('type', seq(':', $._expression)),

    explicit: $ => seq('@', prec(PREC.max, $._term)),
    _binder_ident: $ => choice($.identifier, $.hole),
    _binder_default: $ => field('default', seq(':=', $._expression)),
    explicit_binder: $ => seq(
      '(',
      field('name', repeat1($._binder_ident)),
      field('type', optional($._type_spec)),
      optional(choice($._binder_default)),
      ')',
    ),
    implicit_binder: $ => seq(
      '{',
      field('name', repeat1($._binder_ident)),
      field('type', optional($._type_spec)),
      '}',
    ),
    instance_binder: $ => seq(
      '[',
      optional(seq(field('name', $.identifier), ':')),
      field('type', $._expression),
      ']',
    ),
    _bracketed_binder: $ => choice(
      $.explicit_binder,
      $.implicit_binder,
      $.instance_binder,
    ),
    proj: $ => seq(
      field('term', prec(PREC.lead, $._term)),
      token.immediate('.'),
      field('name', choice($.identifier, $.number)),
    ),
    arrow: $ => prec.right(25, seq($._term, choice('->', '→'), $._expression)),

    forall: $ => prec(-4, seq(
      '∀',
      field(
        'binders',
        alias(repeat1(choice($._simple_binder, $._bracketed_binder)), $.binders),
      ),
      ',',
      field('body', $._expression),
    )),

    _simple_binder: $ => seq(
      field('name', repeat1($._binder_ident)),
      optional($._type_spec),
    ),

    match_alt: $ => match_alt($),
    _match_alts: $ => repeat1($.match_alt),
    _match_discr: $ => seq(
      optional(seq($.identifier, token.immediate(':'))),
      $._expression,
    ),

    true: $ => 'true',
    false: $ => 'false',

    match: $ => prec.left(seq(
      'match',
      field('value', sep1($._match_discr, ', ')),
      field('type', optional($._type_spec)),
      'with',
      field('patterns', $._match_alts),
    )),

    borrowed: $ => seq('@&', prec(PREC.lead, $._term)),
    quoted_name: $ => seq('`', $.identifier),
    double_quoted_name: $ => seq('``', $.identifier),

    _simple_binder_without_type: $ => repeat1($._binder_ident),
    _let_id_lhs: $ => seq(
      field('name', $.identifier),
      field('binders', repeat(
        choice($._simple_binder_without_type, $._bracketed_binder)),
      ),
      field('type', optional($._type_spec)),
    ),
    _let_id_decl: $ => seq($._let_id_lhs, ":=", field('body', $._expression)),
    _let_pattern_decl: $ => seq($._term, optional($._type_spec), ":=", $._expression),
    _let_equations_decl: $ => seq($._let_id_lhs, field('body', $._match_alts)),
    _let_decl: $ => choice(
      $._let_id_decl, $._let_pattern_decl, $._let_equations_decl,
    ),

    _have_id_lhs: $ => min1(
      seq(
        field('name', $.identifier),
        field('binders', repeat(
          choice($._simple_binder_without_type, $._bracketed_binder)),
        ),
      ),
      field('type', $._type_spec),
    ),
    _have_id_decl: $ => seq(optional($._have_id_lhs), ':=', $._term),
    _have_eqns_decl: $ => seq(optional($._have_id_lhs), $._match_alts),
    _have_decl: $ => choice(
      $._have_id_decl,
      $._let_pattern_decl,
      $._have_eqns_decl,
    ),
    have: $ => prec(PREC.lead,
      seq('have', $._have_decl, optional(';'), $._term),
    ),
    _attr_kind: $ => choice('scoped', 'local'),
    _attr_instance: $ => seq(optional($._attr_kind), $._attribute),
    attributes: $ => seq('@[', sep1($._attr_instance, ','), ']'),
    _let_rec_decl: $ => seq($._let_decl),

    _where_decls: $ => seq(
      'where', repeat1(seq(alias($._let_rec_decl, $.where_decl))),
    ),
    _match_alts_where_decls: $ => seq($._match_alts, optional($._where_decls)),

    named_argument: $ => seq(
      '(', $.identifier, ':=', $._expression, ')',
    ),
    ellipsis: $ => '..',
    _argument: $ => choice(
      $.named_argument,
      $.ellipsis,
      prec(PREC.arg, $._expression),
    ),

    quoted_tactic: $ => seq('`(tactic|', $._tactic, ')'),

    _term: $ => term.all($),

    // src/Lean/Parser/Do.lean
    lift_method: $ => prec(PREC.min, seq($._left_arrow, $._term)),

    _do_term: $ => choice(
      $.lift_method,
    ),

    // src/Init/Notation.lean
    product: $ => prec.right(35, seq($._term, '×', $._term)),

    unary_expression: $ => prec(PREC.max, seq(
      choice('¬', '!'),
      prec(40, $._term),
    )),

    subtype: $ => seq(
      '{',
      $.identifier,
      optional(seq(':', $._term)),
      '//',
      $._expression,
      '}',
    ),

    list: $ => seq('[', sep0($._expression, ','), ']'),

    assumption_literal: $ => seq('‹', $._term, '›'),
    if_then_else: $ => seq(
      'if',
      $._expression,
      'then',
      $._expression,
      'else',
      $._expression,
    ),

    _notation_term: $ => choice(
      $.product,
      $.unary_expression,
      $.subtype,
      $.if_then_else,
      $.list,
      $.assumption_literal,
    ),

    // src/Init/NotationExtra.lean
    exists: $ => prec(-4, seq(
      '∃',
      field(
        'binders',
        alias(repeat1(choice($._simple_binder, $._bracketed_binder)), $.binders),
      ),
      ',',
      field('body', $._expression),
    )),

    _notation_extra_term: $ => choice(
      $.exists,
    ),

    // src/Init/Data/Array/Basic.lean
    array: $ => seq('#[', sep0($._expression, ','), ']'),

    // src/Init/Data/Array/Subarray.lean
    subarray: $ => prec(PREC.max, seq(
      field('term', $._expression),
      token.immediate('['),
      choice(
        field('value', $._expression),
        seq(
          optional(field('start', $._expression)),
          ':',
          field('stop', $._expression),
        ),
        seq(
          field('start', $._expression),
          ':',
          optional(field('stop', $._expression)),
        ),
      ),
      ']',
    )),

    // src/Init/Data/Range.lean
    range: $ => seq(
      '[',
      optional(field('start', $._expression)),
      ':',
      field('stop', $._expression),
      optional(seq(
        ':',
        field('step', $._expression),
      )),
      ']',
    ),

    // src/Init/Data/ToString/Macro.lean
    interpolated_string: $ => seq(
      's!"',
      repeat(choice(/[^"]/, $.quoted_char, $.interpolation)),
      '"',
    ),
    interpolation: $ => seq(
      '{', $._expression, '}'
    ),
  },
}
