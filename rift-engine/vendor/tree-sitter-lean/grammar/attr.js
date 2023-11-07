// src/Lean/Parser/Attr.lean

module.exports = {
  _attribute: $ => choice(
    $._simple_attribute,
    $._extern_attribute,
  ),

  _simple_attribute: $ => field(
    'name', seq($.identifier, optional($.identifier)),
  ),

  _extern_entry: $ => seq(
    optional($.identifier),
    optional('inline'),
    field('extern', $.string),
  ),
  _extern_attribute: $ => seq(
    'extern',
    optional($.number),
    repeat($._extern_entry),
  )
}
