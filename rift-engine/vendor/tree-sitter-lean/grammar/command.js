const do_ = require('./do.js')
const {min1, sep1} = require('./util.js')

// src/Lean/Parser/Command.lean
module.exports = {
  _visibility: $ => choice('private', 'protected'),
  _decl_modifiers: $ => seq(
    min1(
      $.attributes,
      repeat1(
        choice('noncomputable', 'partial', $._visibility, 'unsafe'),
      ),
    ),
  ),
  _decl_id: $ => field('name', prec(2, $.identifier)),
  _decl_sig: $ => seq(
    alias(
      repeat(prec.left(choice($._simple_binder_without_type, $._bracketed_binder))),
      $.binders,
    ),
    $._type_spec,
  ),
  _opt_decl_sig: $ => prec.left(min1(
    alias(
      repeat1(prec.left(choice($._simple_binder_without_type, $._bracketed_binder))),
      $.binders,
    ),
    $._type_spec,
  )),
  _decl_val_simple: $ => seq(':=', $._expression),
  _decl_val_equations: $ => $._match_alts_where_decls,
  _decl_val: $ => field('body', choice(
    $._decl_val_simple,
    $._decl_val_equations,
    $._where_decls,
  )),
  abbrev: $ => seq(
    'abbrev',
    $._decl_id,
    optional($._opt_decl_sig),
    $._decl_val,
  ),
  def: $ => seq(
    'def',
    $._decl_id,
    optional($._opt_decl_sig),
    $._decl_val,
  ),
  theorem: $ => seq(
    'theorem',
    $._decl_id,
    $._decl_sig,
    $._decl_val,
  ),
  constant: $ => seq(
    'constant',
    $._decl_id,
    $._decl_sig,
    optional($._decl_val_simple),
  ),
  instance: $ => seq(
    'instance',
    optional($._decl_id),
    $._decl_sig,
    $._decl_val,
  ),
  axiom: $ => seq('axiom', $._decl_id, $._decl_sig),
  example: $ => seq(
    'example',
    $._decl_sig,
    $._decl_val,
  ),
  constructor: $ => seq(
    '|',
    field('name', $.identifier),
    optional($._opt_decl_sig),
  ),
  _deriving: $ => field('deriving', seq('deriving', sep1($.identifier, ','))),
  inductive: $ => seq(
    'inductive',
    $._decl_id,
    optional($._opt_decl_sig),
    optional(choice(':=', 'where')),
    optional(field('constructors', repeat1($.constructor))),
    optional($._deriving),
  ),
  class_inductive: $ => seq(
    'class', 'inductive',
    $._decl_id,
    optional($._opt_decl_sig),
    optional(choice(':=', 'where')),
    optional(field('constructors', repeat1($.constructor))),
    optional($._deriving),
  ),
  _struct_explicit_binder: $ => seq(
    '(',
    field('name', repeat1($.identifier)),
    field('type', optional($._opt_decl_sig)),
    optional($._binder_default),
    ')',
  ),
  _struct_implicit_binder: $ => seq(
    '{',
    field('name', repeat1($.identifier)),
    field('type', $._decl_sig),
    '}',
  ),
  _struct_instance_binder: $ => seq(
    '[',
    field('name', repeat1($.identifier)),
    field('type', $._decl_sig),
    ']',
  ),
  _struct_simple_binder: $ => prec.left(seq(
    field('name', $.identifier),
    field('type', optional($._opt_decl_sig)),
    optional($._binder_default),
  )),
  _struct_field: $ => alias(
    choice(
      alias($._struct_explicit_binder, $.explicit_binder),
      alias($._struct_implicit_binder, $.implicit_binder),
      alias($._struct_instance_binder, $.instance_binder),
      $._struct_simple_binder,
    ), $.field,
  ),
  _struct_constructor: $ => seq($.identifier, '::'),
  _extends: $ => field('extends', seq('extends', sep1($._expression, ','))),
  structure: $ => seq(
    choice('structure', 'class'),
    $._decl_id,
    alias(repeat($._bracketed_binder), $.binders),
    optional($._extends),
    optional($._type_spec),
    optional(
      seq(
        choice(':=', 'where'),
        optional($._struct_constructor),
        field('fields', repeat($._struct_field)),
      ),
    ),
    optional($._deriving),
  ),
  declaration: $ => seq(
    optional($._decl_modifiers),
    choice(
      $.abbrev,
      $.def,
      $.theorem,
      $.constant,
      $.instance,
      $.axiom,
      $.example,
      $.inductive,
      $.class_inductive,
      $.structure,
    ),
  ),
  section: $ => seq(
    'section',
    optional(field('name', $.identifier)),
    field('body', repeat($._command)),
    'end',
    optional($.identifier),
  ),
  namespace: $ => seq(
    'namespace',
    field('name', $.identifier),
    field('body', repeat($._command)),
    'end',
    $.identifier,
  ),
  variable: $ => seq('variable', repeat1($._bracketed_binder)),
  universe: $ => seq('universe', repeat1($.identifier)),
  hash_command: $ => seq(
    choice('#check', '#check_failure', '#eval', '#print', '#reduce'),
    $._expression,
  ),

  attribute: $ => seq(
    'attribute',
    '[',
    sep1(choice($._attribute, seq("-", $._attribute)), ','),
    ']',
    field('term', $.identifier),
  ),
  export: $ => seq(
    'export',
    field('class', $.identifier),
    '(',
    repeat1($.identifier),
    ')',
  ),

  open: $ => seq(
    'open',
    repeat1(field('namespace', $.identifier)),
    optional(seq('in', $._command)),
  ),

  // PERFIXME: quoted causes ~2x performance slowdown :/
  quoted: $ => seq('`(', choice($._term, repeat1($._command)), ')'),
  _command_term: $ => choice(
    $.quoted,
  ),

  builtin_initialize: $ => seq(
    optional($._visibility),
    'builtin_initialize',
    optional(seq($.identifier, $._type_spec, $._left_arrow)),
    $._do_seq,
  ),

  _command: $ => choice(
    $.declaration,
    $.section,
    $.namespace,
    $.variable,
    $.universe,
    $.hash_command,
    $.attribute,
    $.export,
    $.open,
    $.builtin_initialize,

    // src/Lean/Parser/Syntax.lean
    $.mixfix,
    $.notation,
    $.macro_rules,
    $.syntax,
    $.macro,
    $.elab,
  ),
}
