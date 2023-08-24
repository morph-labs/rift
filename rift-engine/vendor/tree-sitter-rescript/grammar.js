module.exports = grammar({
  name: 'rescript',

  externals: $ => [
    $._newline,
    $.comment,
    $._newline_and_comment,
    '"',
    "`",
    $._template_chars,
    $._lparen,
    $._rparen,
    $._list_constructor,
    $._decorator,
    $._decorator_inline,
  ],

  extras: $ => [
    $.comment,
    $.decorator,
    /[\s\uFEFF\u2060\u200B\u00A0]/
  ],

  supertypes: $ => [
    $.statement,
    $.declaration,
    $.expression,
    $.primary_expression,
    $._type,
    $.module_expression,
    $.module_primary_expression,
  ],

  inline: $ => [
    $._module_definition,
  ],

  precedences: $ => [
    // + - Operators -> precendence
    [
      'unary_not',
      'member',
      'call',
      $.spread_element,
      $.await_expression,
      $.pipe_expression,
      $.lazy_expression,
      'binary_times',
      'binary_pow',
      'binary_plus',
      'binary_compare',
      'binary_relation',
      'binary_and',
      'binary_or',
      'coercion_relation',
      $.expression,
      $.primary_expression,
      $.ternary_expression,
      $.mutation_expression,
      $.function,
      $.let_declaration,
    ],
    // Nested.Module.Path precendence
    [
      $.module_primary_expression,
      $.value_identifier_path,
      $.nested_variant_identifier,
      $.module_identifier_path,
    ],
    [$._jsx_attribute_value, $.pipe_expression],
    [$.function_type_parameters, $.function_type],
    [$._reserved_identifier, $.module_unpack],
    [$.lazy_pattern, $.or_pattern]
  ],

  conflicts: $ => [
    [$.unit, $.formal_parameters],
    [$.primary_expression, $._pattern],
    [$.primary_expression, $.record_pattern],
    [$.primary_expression, $.spread_pattern],
    [$.primary_expression, $._literal_pattern],
    [$.primary_expression, $.lazy_pattern],
    [$.primary_expression, $._jsx_child],
    [$.tuple_type, $.function_type_parameter],
    [$.list, $.list_pattern],
    [$.array, $.array_pattern],
    [$.type_declaration],
    [$.let_declaration],
    [$.variant_identifier, $.module_identifier],
    [$.variant, $.variant_pattern],
    [$.variant_arguments, $._variant_pattern_parameters],
    [$.polyvar, $.polyvar_pattern],
    [$._pattern],
    [$._record_element, $._record_single_field],
    [$._record_pun_field, $._record_single_pun_field],
    [$._record_field_name, $.record_pattern],
    [$._statement, $._one_or_more_statements],
    [$._inline_type, $.function_type_parameters],
    [$.primary_expression, $.parameter, $._pattern],
    [$.parameter, $._pattern],
    [$.parameter, $.parenthesized_pattern],
    [$.parameter, $.tuple_item_pattern],
    [$.unit, $._function_type_parameter_list],
    [$.functor_parameter, $.module_primary_expression, $.module_identifier_path],
    [$._reserved_identifier, $.function],
    [$.exception_pattern, $.or_pattern],
    [$.type_binding, $._inline_type],
    [$._module_structure, $.parenthesized_module_expression]
  ],

  rules: {
    source_file: $ => seq(
      repeat($._statement_delimeter),
      repeat($._statement)
    ),

    _statement: $ => seq(
      $.statement,
      repeat1($._statement_delimeter)
    ),

    _statement_delimeter: $ => choice(
      ';',
      $._newline,
      alias($._newline_and_comment, $.comment),
    ),

    _one_or_more_statements: $ => seq(
      repeat($._statement),
      $.statement,
      optional($._statement_delimeter),
    ),

    statement: $ => choice(
      $.expression_statement,
      $.declaration,
      $.open_statement,
      $.include_statement,
    ),

    block: $ => prec.right(seq(
      '{',
      optional($._one_or_more_statements),
      '}',
    )),

    open_statement: $ => seq(
      'open',
      optional('!'),
      $.module_expression,
    ),

    include_statement: $ => seq(
      'include',
      choice(
        $._module_definition,
        parenthesize($._module_structure)
      )
    ),

    declaration: $ => choice(
      $.type_declaration,
      $.let_declaration,
      $.module_declaration,
      $.external_declaration,
      $.exception_declaration,
    ),

    module_binding: $ => prec.left(seq(
      field('name', choice($.module_identifier, $.type_identifier)),
      optional(seq(
        ':',
        field('signature', choice($.block, $.module_expression, $.functor)),
      )),
      optional(seq(
        '=',
        field('definition', $._module_definition),
      )),
    )),

    module_declaration: $ => seq(
      'module',
      optional('rec'),
      optional('type'),
      sep1('and', $.module_binding)
    ),

    _module_structure: $ => seq(
      $._module_definition,
      optional($.module_type_annotation),
    ),

    _module_definition: $ => choice(
      $.block,
      $.module_expression,
      $.functor,
      $.extension_expression,
    ),

    module_unpack: $ => seq(
      'unpack',
      '(',
      choice(
        seq(
          choice($.value_identifier, $.value_identifier_path, $.member_expression),
          optional($.module_type_annotation)
        ),
        $.call_expression,
        $.extension_expression
      ),
      ')'
    ),

    functor: $ => seq(
      field('parameters', $.functor_parameters),
      optional(field('return_module_type', $.module_type_annotation)),
      '=>',
      field('body', $._module_definition),
    ),

    functor_parameters: $ => seq(
      '(',
      optional(commaSep1t($.functor_parameter)),
      ')',
    ),

    functor_parameter: $ => seq(
      $.module_identifier,
      optional($.module_type_annotation),
    ),

    module_type_annotation: $ => seq(
      ':',
      choice(
        $.module_expression,
        $.block,
      )
    ),

    external_declaration: $ => seq(
      'external',
      $.value_identifier,
      $.type_annotation,
      '=',
      $.string,
    ),

    exception_declaration: $ => seq(
      'exception',
      $.variant_identifier,
      optional($.variant_parameters),
      optional(seq('=', choice($.variant_identifier, $.nested_variant_identifier)))
    ),

    type_declaration: $ => seq(
      optional('export'),
      'type',
      optional('rec'),
      sep1(
        seq(repeat($._newline), 'and'),
        $.type_binding
      )
    ),

    type_binding: $ => seq(
      field('name', choice($.type_identifier, $.type_identifier_path)),
      optional($.type_parameters),
      optional(seq(
        optional(seq('=', $._non_function_inline_type)),
        optional(seq(
          choice('=', '+='),
          optional('private'),
          field('body', $._type),
        )),
        repeat($.type_constraint),
      )),
    ),

    type_parameters: $ => seq(
      '<',
      commaSep1t(
        seq(
          optional(choice('+', '-')),
          $.type_identifier
        )
      ),
      '>',
    ),

    type_annotation: $ => seq(
      ':',
      $._inline_type,
    ),

    _type: $ => choice(
      $._inline_type,
      $.variant_type,
      $.record_type,
      $.as_aliasing_type,
    ),

    _inline_type: $ => choice(
      $._non_function_inline_type,
      $.function_type,
    ),

    _non_function_inline_type: $ => choice(
      $._type_identifier,
      $.tuple_type,
      $.polyvar_type,
      $.object_type,
      $.generic_type,
      $.unit_type,
      $.module_pack,
      $.unit,
      $.polymorphic_type,
    ),

    polymorphic_type: $ => seq(
      choice(repeat1($.type_identifier), $.abstract_type),
      '.',
      $._inline_type
    ),

    type_constraint: $ => seq(
      'constraint',
      $._type,
      '=',
      $._type
    ),

    tuple_type: $ => prec.dynamic(-1, seq(
      '(',
      commaSep1t($._type),
      ')',
    )),

    variant_type: $ => prec.left(seq(
      optional('|'),
      barSep1($.variant_declaration),
    )),

    variant_declaration: $ => prec.right(seq(
      $.variant_identifier,
      optional($.variant_parameters),
      optional($.type_annotation),
    )),

    variant_parameters: $ => seq(
      '(',
      commaSep1t($._type),
      ')',
    ),

    polyvar_type: $ => prec.left(seq(
      choice('[', '[>', '[<',),
      optional('|'),
      barSep1($.polyvar_declaration),
      ']',
    )),

    polyvar_declaration: $ => prec.right(
      choice(
        seq(
          $.polyvar_identifier,
          optional($.polyvar_parameters),
        ),
        $._inline_type
      )
    ),

    polyvar_parameters: $ => seq(
      '(',
      commaSep1t($._type),
      ')',
    ),

    record_type: $ => seq(
      '{',
      commaSept($.record_type_field),
      '}',
    ),

    record_type_field: $ => seq(
      optional('mutable'),
      alias($.value_identifier, $.property_identifier),
      optional('?'),
      $.type_annotation,
    ),

    object_type: $ => prec.left(seq(
      '{',
      choice(
        commaSep1t($._object_type_field),
        seq('.', commaSept($._object_type_field)),
        seq('..', commaSept($._object_type_field)),
      ),
      '}',
    )),

    _object_type_field: $ => alias($.object_type_field, $.field),

    object_type_field: $ => choice(
      seq('...', choice($.type_identifier, $.type_identifier_path)),
      seq(
        alias($.string, $.property_identifier),
        ':',
        $._type,
      ),

    ),

    generic_type: $ => prec.left(seq(
      $._type_identifier,
      $.type_arguments,
    )),

    type_arguments: $ => seq(
      '<',
      commaSep1t($._type),
      '>'
    ),

    function_type: $ => prec.left(seq(
      $.function_type_parameters,
      '=>',
      $._type,
    )),

    function_type_parameters: $ => choice(
      $._non_function_inline_type,
      $._function_type_parameter_list,
    ),

    _function_type_parameter_list: $ => seq(
      '(',
      commaSept(alias($.function_type_parameter, $.parameter)),
      ')',
    ),

    function_type_parameter: $ => seq(
      optional($.uncurry),
      choice(
        $._type,
        seq($.uncurry, $._type),
        $.labeled_parameter,
      ),
    ),

    let_declaration: $ => seq(
      choice('export', 'let'),
      optional('rec'),
      sep1(
        seq(repeat($._newline), 'and'),
        $.let_binding
      )
    ),

    let_binding: $ => seq(
      field('pattern', $._pattern),
      choice(
        seq(
          $.type_annotation,
          optional(
            seq('=',
              field('body', $.expression)
            )
          )
        ),
        seq(
          '=',
          field('body', $.expression),
        )
      )
    ),

    expression_statement: $ => $.expression,

    expression: $ => choice(
      $.primary_expression,
      $._jsx_element,
      $.jsx_fragment,
      $.unary_expression,
      $.binary_expression,
      $.coercion_expression,
      $.ternary_expression,
      $.for_expression,
      $.while_expression,
      $.mutation_expression,
      $.await_expression,
      $.block,
      $.assert_expression,
    ),

    primary_expression: $ => choice(
      $.parenthesized_expression,
      $.value_identifier_path,
      $.value_identifier,
      $.number,
      $.string,
      $.template_string,
      $.character,
      $.true,
      $.false,
      $.function,
      $.unit,
      $.record,
      $.object,
      $.tuple,
      $.array,
      $.list,
      $.variant,
      $.polyvar,
      $.if_expression,
      $.switch_expression,
      $.try_expression,
      $.call_expression,
      $.pipe_expression,
      $.subscript_expression,
      $.member_expression,
      $.module_pack,
      $.extension_expression,
      $.lazy_expression,
    ),

    parenthesized_expression: $ => seq(
      '(',
      $.expression,
      optional($.type_annotation),
      ')'
    ),

    value_identifier_path: $ => seq(
      $.module_primary_expression,
      '.',
      $.value_identifier,
    ),

    function: $ => prec.left(seq(
      optional('async'),
      choice(
        field('parameter', $.value_identifier),
        $._definition_signature
      ),
      '=>',
      field('body', $.expression),
    )),

    record: $ => seq(
      '{',
      choice(
        $._record_single_field,
        $._record_single_pun_field,
        commaSep2t($._record_element),
      ),
      '}',
    ),

    _record_element: $ => choice(
      $.spread_element,
      $.record_field,
      alias($._record_pun_field, $.record_field),
    ),

    record_field: $ => seq(
      $._record_field_name,
      ':',
      optional('?'),
      $.expression,
    ),

    _record_pun_field: $ => seq(
      optional('?'),
      $._record_field_name,
    ),

    _record_single_field: $ => seq(
      $.record_field,
      optional(','),
    ),

    _record_single_pun_field: $ => seq(
      '?',
      $._record_field_name,
      optional(','),
    ),

    _record_field_name: $ => choice(
      alias($.value_identifier, $.property_identifier),
      alias($.value_identifier_path, $.property_identifier),
    ),

    object: $ => seq(
      '{',
      choice(
        commaSep1t($._object_field),
        seq('.', commaSept($._object_field)),
        seq('..', commaSept($._object_field)),
      ),
      '}',
    ),

    _object_field: $ => alias($.object_field, $.field),

    object_field: $ => seq(
      alias($.string, $.property_identifier),
      ':',
      $.expression,
    ),

    tuple: $ => seq(
      '(',
      commaSep2t(
        $.expression
      ),
      ')',
    ),

    array: $ => seq(
      '[',
      commaSept($.expression),
      ']'
    ),

    list: $ => seq(
      $._list_constructor,
      '{',
      optional(commaSep1t($._list_element)),
      '}'
    ),

    _list_element: $ => choice(
      $.expression,
      $.spread_element,
    ),

    if_expression: $ => seq(
      'if',
      $.expression,
      $.block,
      repeat($.else_if_clause),
      optional($.else_clause),
    ),

    else_if_clause: $ => seq(
      'else',
      'if',
      $.expression,
      $.block,
    ),

    else_clause: $ => seq(
      'else',
      $.block,
    ),

    switch_expression: $ => seq(
      'switch',
      $.expression,
      '{',
      repeat($.switch_match),
      '}',
    ),

    switch_match: $ => prec.dynamic(-1, seq(
      '|',
      field('pattern', $._pattern),
      optional($.guard),
      '=>',
      field('body', alias($._one_or_more_statements, $.sequence_expression)),
    )),

    guard: $ => seq(
      choice('if', 'when'),
      $.expression,
    ),

    polyvar_type_pattern: $ => seq(
      '#',
      '...',
      $._type_identifier,
    ),

    try_expression: $ => seq(
      'try',
      $.expression,
      'catch',
      '{',
      repeat($.switch_match),
      '}',
    ),

    as_aliasing: $ => prec.left(seq(
      'as',
      $._pattern,
      optional($.type_annotation)
    )),

    as_aliasing_type: $ => seq($._type, 'as', $.type_identifier),

    assert_expression: $ => prec.left(seq('assert', $.expression)),

    call_expression: $ => prec('call', seq(
      field('function', $.primary_expression),
      field('arguments', alias($.call_arguments, $.arguments)),
    )),

    pipe_expression: $ => prec.left(seq(
      choice(
        $.primary_expression,
        $.block,
      ),
      choice('->', '|>'),
      choice(
        $.primary_expression,
        $.block,
      ),
    )),

    module_pack: $ => seq(
      'module',
      parenthesize(choice($._module_structure, $.type_identifier_path))
    ),

    call_arguments: $ => seq(
      '(',
      optional($.uncurry),
      optional(commaSep1t($._call_argument)),
      ')'
    ),

    _call_argument: $ => choice(
      seq(
        $.expression,
        optional($.type_annotation),
      ),
      $.labeled_argument,
    ),

    labeled_argument: $ => seq(
      '~',
      field('label', $.value_identifier),
      optional(choice(
        '?',
        seq(
          '=',
          optional('?'),
          field('value', $.expression),
          optional(field('type', $.type_annotation)),
        ),
      )),
    ),

    _definition_signature: $ => seq(
      field('parameters', $.formal_parameters),
      optional(field('return_type', alias($._return_type_annotation, $.type_annotation))),
    ),

    _return_type_annotation: $ => seq(
      ':',
      $._non_function_inline_type,
    ),

    formal_parameters: $ => seq(
      '(',
      optional(commaSep1t($.parameter)),
      ')'
    ),

    parameter: $ => seq(
      optional($.uncurry),
      choice(
        seq($._pattern, optional($.type_annotation)),
        $.labeled_parameter,
        $.unit,
        $.abstract_type
      ),
    ),

    labeled_parameter: $ => seq(
      '~',
      $.value_identifier,
      optional($.as_aliasing),
      optional($.type_annotation),
      optional(field('default_value', $._labeled_parameter_default_value)),
    ),

    abstract_type: $ => seq(
      'type',
      repeat1($.type_identifier),
    ),

    _labeled_parameter_default_value: $ => seq(
      '=',
      choice(
        '?',
        $.expression,
      ),
    ),

    // This negative dynamic precedence ensures that during error recovery,
    // unfinished constructs are generally treated as literal expressions,
    // not patterns.
    _pattern: $ => prec.dynamic(-1, seq(
      choice(
        $.value_identifier,
        $._literal_pattern,
        $._destructuring_pattern,
        $.polyvar_type_pattern,
        $.unit,
        $.module_pack,
        $.lazy_pattern,
        $.parenthesized_pattern,
        $.or_pattern,
        $.range_pattern,
        $.exception_pattern
      ),
      optional($.as_aliasing),
    )),

    parenthesized_pattern: $ => seq('(', $._pattern, optional($.type_annotation), ')'),

    range_pattern: $ => seq(
      $._literal_pattern,
      '..',
      $._literal_pattern,
    ),

    or_pattern: $ => prec.left(seq($._pattern, '|', $._pattern)),

    exception_pattern: $ => seq('exception', $._pattern),

    _destructuring_pattern: $ => choice(
      $.variant_pattern,
      $.polyvar_pattern,
      $.record_pattern,
      $.tuple_pattern,
      $.array_pattern,
      $.list_pattern,
    ),

    _literal_pattern: $ => choice(
      $.string,
      $.template_string,
      $.character,
      $.number,
      $.true,
      $.false,
    ),

    variant_pattern: $ => seq(
      optional('?'),
      choice(
        $.variant_identifier,
        $.nested_variant_identifier,
      ),
      optional(alias($._variant_pattern_parameters, $.formal_parameters))
    ),

    _variant_pattern_parameters: $ => seq(
      '(',
      commaSept($._variant_pattern_parameter),
      ')',
    ),

    _variant_pattern_parameter: $ => seq(
      $._pattern,
      optional($.type_annotation),
    ),

    polyvar_pattern: $ => seq(
      $.polyvar_identifier,
      optional(alias($._variant_pattern_parameters, $.formal_parameters))
    ),

    record_pattern: $ => seq(
      '{',
      commaSep1t(seq(
        choice(
          $.value_identifier,
          $.value_identifier_path,
        ),
        optional(seq(
          ':',
          $._pattern,
        )),
      )),
      '}'
    ),

    tuple_item_pattern: $ => seq(
      $._pattern,
      optional($.type_annotation),
    ),

    tuple_pattern: $ => seq(
      '(',
      commaSep2t($.tuple_item_pattern),
      ')',
    ),

    array_pattern: $ => seq(
      '[',
      optional(commaSep1t(
        $._collection_element_pattern
      )),
      ']',
    ),

    list_pattern: $ => seq(
      $._list_constructor,
      '{',
      optional(commaSep1t($._collection_element_pattern)),
      '}',
    ),

    _collection_element_pattern: $ => seq(
      choice($._pattern, $.spread_pattern),
      optional($.as_aliasing)
    ),

    spread_pattern: $ => seq(
      '...',
      choice($.value_identifier, $.list_pattern, $.array_pattern),
    ),

    lazy_pattern: $ => seq(
      'lazy',
      $._pattern
    ),

    _jsx_element: $ => choice($.jsx_element, $.jsx_self_closing_element),

    jsx_element: $ => seq(
      field('open_tag', $.jsx_opening_element),
      repeat($._jsx_child),
      field('close_tag', $.jsx_closing_element)
    ),

    jsx_fragment: $ => seq('<', '>', repeat($._jsx_child), '<', '/', '>'),

    jsx_expression: $ => seq(
      '{',
      optional(choice(
        $._one_or_more_statements,
        $.spread_element
      )),
      '}'
    ),

    _jsx_child: $ => choice(
      $.value_identifier,
      $.value_identifier_path,
      $.number,
      $.string,
      $.template_string,
      $.character,
      $._jsx_element,
      $.jsx_fragment,
      $.block,
      $.spread_element,
      $.member_expression
    ),

    jsx_opening_element: $ => prec.dynamic(-1, seq(
      '<',
      field('name', $._jsx_element_name),
      repeat(field('attribute', $._jsx_attribute)),
      '>'
    )),

    _jsx_identifier: $ => alias(
      choice($.value_identifier, $.module_identifier),
      $.jsx_identifier
    ),

    nested_jsx_identifier: $ => prec('member', seq(
      choice($._jsx_identifier, $.nested_jsx_identifier),
      '.',
      $._jsx_identifier
    )),

    _jsx_element_name: $ => choice(
      $._jsx_identifier,
      $.nested_jsx_identifier,
    ),

    jsx_closing_element: $ => seq(
      '<',
      '/',
      field('name', $._jsx_element_name),
      '>'
    ),

    jsx_self_closing_element: $ => seq(
      '<',
      field('name', $._jsx_element_name),
      repeat(field('attribute', $._jsx_attribute)),
      '/',
      '>'
    ),

    _jsx_attribute_name: $ => alias($.value_identifier, $.property_identifier),

    _jsx_attribute: $ => choice($.jsx_attribute, $.jsx_expression),

    jsx_attribute: $ => seq(
      optional('?'),
      $._jsx_attribute_name,
      optional(seq(
        '=',
        optional('?'),
        $._jsx_attribute_value
      )),
    ),

    _jsx_attribute_value: $ => choice(
      $.primary_expression,
      $.jsx_expression,
    ),

    mutation_expression: $ => seq(
      $._mutation_lvalue,
      choice('=', ':='),
      $.expression,
    ),

    _mutation_lvalue: $ => choice(
      $.value_identifier,
      $.member_expression,
      $.subscript_expression,
    ),

    await_expression: $ => seq(
      'await',
      $.expression,
    ),

    decorator: $ => choice(
      alias($._decorator_inline, $.decorator_identifier),
      seq(alias($._decorator, $.decorator_identifier), $.decorator_arguments)
    ),

    decorator_arguments: $ => seq(
      '(',
      choice(
        commaSept($.expression),
        $.type_annotation
      ),
      ')',
    ),

    subscript_expression: $ => prec.right('member', seq(
      field('object', $.primary_expression),
      '[', field('index', $.expression), ']'
    )),

    member_expression: $ => prec('member', seq(
      field('record', $.primary_expression),
      '.',
      optional(seq(
        field('module', $.module_identifier),
        '.'
      )),
      field('property', alias($.value_identifier, $.property_identifier)),
    )),

    spread_element: $ => seq('...', $.expression),

    ternary_expression: $ => prec.left(seq(
      field('condition', $.expression),
      '?',
      field('consequence', $.expression),
      ':',
      field('alternative', $.expression)
    )),

    for_expression: $ => seq(
      'for',
      $.value_identifier,
      'in',
      $.expression,
      choice('to', 'downto'),
      $.expression,
      $.block,
    ),

    while_expression: $ => seq(
      'while',
      $.expression,
      $.block,
    ),

    lazy_expression: $ => seq(
      'lazy',
      $.expression,
    ),

    binary_expression: $ => choice(
      ...[
        ['&&', 'binary_and'],
        ['||', 'binary_or'],
        ['++', 'binary_plus'],
        ['+', 'binary_plus'],
        ['+.', 'binary_plus'],
        ['-', 'binary_plus'],
        ['-.', 'binary_plus'],
        ['*', 'binary_times'],
        ['*.', 'binary_times'],
        ['**', 'binary_pow'],
        ['/', 'binary_times'],
        ['/.', 'binary_times'],
        ['<', 'binary_relation'],
        ['<=', 'binary_relation'],
        ['==', 'binary_relation'],
        ['===', 'binary_relation'],
        ['!=', 'binary_relation'],
        ['!==', 'binary_relation'],
        ['>=', 'binary_relation'],
        ['>', 'binary_relation'],
      ].map(([operator, precedence]) =>
        prec.left(precedence, seq(
          field('left', $.expression),
          field('operator', operator),
          field('right', $.expression)
        ))
      )
    ),

    coercion_expression: $ => prec.left(
      'coercion_relation',
      seq(
        field('left', $.expression),
        field('operator', ':>'),
        field('right', $._type_identifier),
      )
    ),

    unary_expression: $ => choice(...[
      ['!', 'unary_not'],
      ['-', 'unary_not'],
      ['-.', 'unary_not'],
      ['+', 'unary_not'],
      ['+.', 'unary_not'],
    ].map(([operator, precedence]) =>
      prec.left(precedence, seq(
        field('operator', operator),
        field('argument', $.expression)
      ))
    )),

    extension_expression: $ => prec.right(seq(
      repeat1('%'),
      $.extension_identifier,
      optional(
        $._extension_expression_payload,
      )
    )),

    _extension_expression_payload: $ => seq(
      '(',
      $._one_or_more_statements,
      // explicit newline here because it won’t be reported otherwise by the scanner
      // because we’re in parens
      optional($._newline),
      ')',
    ),

    variant: $ => prec.right(seq(
      choice($.variant_identifier, $.nested_variant_identifier),
      optional(alias($.variant_arguments, $.arguments)),
    )),

    nested_variant_identifier: $ => seq(
      $.module_primary_expression,
      '.',
      $.variant_identifier
    ),

    variant_arguments: $ => seq(
      '(',
      commaSept(seq(
        $.expression,
        optional($.type_annotation),
      )),
      ')',
    ),

    polyvar: $ => prec.right(seq(
      $.polyvar_identifier,
      optional(alias($.variant_arguments, $.arguments)),
    )),

    _type_identifier: $ => choice(
      $.type_identifier,
      $.type_identifier_path,
      ".."
    ),

    type_identifier_path: $ => seq(
      $.module_primary_expression,
      '.',
      $.type_identifier
    ),

    module_expression: $ => choice(
      $.module_primary_expression,
      $.module_type_of,
      $.module_type_constraint,
    ),

    module_primary_expression: $ => choice(
      $.parenthesized_module_expression,
      $.module_identifier,
      $.module_identifier_path,
      $.functor_use,
      $.module_unpack,
    ),

    parenthesized_module_expression: $ => seq(
      '(',
      $.module_expression,
      optional($.module_type_annotation),
      ')',
    ),

    module_identifier_path: $ => path(
      $.module_primary_expression,
      $.module_identifier,
    ),

    module_type_of: $ => prec.left(seq(
      'module',
      'type',
      'of',
      choice($.module_expression, $.block)
    )),

    _module_type_constraint_with: $ => prec.right(seq(
      'with',
      sep1(choice('and', 'with'),
        choice($.constrain_module, $.constrain_type)
      ),
    )),

    module_type_constraint: $ => prec.left(choice(
      seq($.module_expression, $._module_type_constraint_with),
      seq(
        '(',
        $.module_expression, $._module_type_constraint_with,
        ')',
        $._module_type_constraint_with
      )
    )),

    constrain_module: $ => seq(
      'module',
      $.module_primary_expression,
      choice('=', ':='),
      $.module_primary_expression,
    ),

    constrain_type: $ => seq(
      'type',
      $._type,
      choice('=', ':='),
      $._type,
    ),

    functor_use: $ => seq(
      $.module_primary_expression,
      alias($.functor_arguments, $.arguments),
    ),

    functor_arguments: $ => seq(
      '(',
      optional(commaSep1t($._functor_argument)),
      ')',
    ),

    _functor_argument: $ => choice(
      $.module_expression,
      $.block,
    ),

    variant_identifier: $ => /[A-Z][a-zA-Z0-9_']*/,

    polyvar_identifier: $ => seq(
      '#',
      choice(
        /[a-zA-Z0-9_']+/,
        seq(
          optional('\\'),
          alias($.string, $.polyvar_string),
        ),
      ),
    ),

    type_identifier: $ => choice(
      /[a-z_'][a-zA-Z0-9_']*/,
      $._escape_identifier,
    ),

    value_identifier: $ => choice(
      /[a-z_][a-zA-Z0-9_']*/,
      $._reserved_identifier,
      $._escape_identifier,
    ),

    _escape_identifier: $ => token(seq('\\"', /[^"]+/, '"')),

    module_identifier: $ => /[A-Z][a-zA-Z0-9_']*/,

    extension_identifier: $ => /[a-zA-Z0-9_\.]+/,

    number: $ => {
      // OCaml: https://github.com/tree-sitter/tree-sitter-ocaml/blob/f1106bf834703f1f2f795da1a3b5f8f40174ffcc/ocaml/grammar.js#L26
      const hex_literal = seq(
        optional(choice('-', '+')),
        /0[xX][0-9A-Fa-f][0-9A-Fa-f_]*(\.[0-9A-Fa-f_]*)?([pP][+\-]?[0-9][0-9_]*)?[g-zG-Z]?/
      )

      const decimal_digits = /\d(_?\d)*/
      const signed_integer = seq(optional(choice('-', '+')), decimal_digits)
      const exponent_part = seq(choice('e', 'E'), signed_integer)

      const binary_literal = seq(choice('0b', '0B'), /[0-1](_?[0-1])*/)

      const octal_literal = seq(choice('0o', '0O'), /[0-7](_?[0-7])*/)

      const bigint_literal = seq(choice(hex_literal, binary_literal, octal_literal, decimal_digits), 'n')

      const decimal_integer_literal = choice(
        repeat('0'),
        seq(repeat('0'), /[1-9]/, optional(seq(optional('_'), decimal_digits)))
      )

      const decimal_literal = seq(
        optional(choice('-', '+')),
        choice(
          seq(decimal_integer_literal, '.', optional(decimal_digits), optional(exponent_part)),
          seq('.', decimal_digits, optional(exponent_part)),
          seq(decimal_integer_literal, exponent_part),
          seq(decimal_digits),
        )
      )

      const int_32_64 = seq(
        optional(choice('-', '+')),
        choice(decimal_integer_literal, binary_literal, octal_literal, hex_literal),
        choice('L', 'l')
      )

      return token(choice(
        hex_literal,
        decimal_literal,
        binary_literal,
        octal_literal,
        bigint_literal,
        int_32_64
      ))
    },

    unit: $ => seq('(', ')'),
    unit_type: $ => 'unit',

    true: $ => 'true',
    false: $ => 'false',

    string: $ => seq(
      '"',
      repeat(choice(
        alias($.unescaped_double_string_fragment, $.string_fragment),
        $.escape_sequence
      )),
      '"'
    ),

    // Workaround to https://github.com/tree-sitter/tree-sitter/issues/1156
    // We give names to the token() constructs containing a regexp
    // so as to obtain a node in the CST.
    //
    unescaped_double_string_fragment: $ =>
      token.immediate(prec(1, /[^"\\]+/)),

    escape_sequence: $ => token.immediate(seq(
      '\\',
      choice(
        /[^xu0-7]/,
        /[0-7]{1,3}/,
        /x[0-9a-fA-F]{2}/,
        /u[0-9a-fA-F]{4}/,
        /u{[0-9a-fA-F]+}/
      )
    )),

    template_string: $ => seq(
      token(seq(
        optional(
          choice(
            /[a-z_][a-zA-Z0-9_']*/,
            // escape_sequence
            seq('\\"', /[^"]+/, '"'),
          )
        ),
        '`',
      )),
      optional($.template_string_content),
      '`'
    ),

    template_string_content: $ =>
      repeat1(
        choice(
          $._template_chars,
          $.template_substitution,
          /\s/,
          choice(
            alias('\\`', $.escape_sequence),
            $.escape_sequence,
          )
        ),
      ),

    template_substitution: $ => choice(
      seq('$', $.value_identifier),
      seq('${', $.expression, '}'),
    ),

    character: $ => seq(
      "'",
      repeat(choice(/[^\\']/, $.escape_sequence)),
      "'"
    ),

    _unescaped_template_string_fragment: $ =>
      token.immediate(prec(1, /[^`\\\$]+/)),

    lparen: $ => alias($._lparen, '('),
    rparen: $ => alias($._rparen, ')'),
    uncurry: $ => '.',

    _reserved_identifier: $ => choice(
      'async',
      'unpack'
    )
  },
});

function barSep1(rule) {
  return seq(rule, repeat(seq('|', rule)));
}

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}

function commaSep2(rule) {
  return seq(rule, ',', commaSep1(rule));
}

function commaSep1t(rule) {
  return seq(commaSep1(rule), optional(','));
}

function commaSep2t(rule) {
  return seq(commaSep2(rule), optional(','));
}

function commaSep(rule) {
  return optional(commaSep1(rule));
}

function commaSept(rule) {
  return optional(commaSep1t(rule));
}

function sep1(delimiter, rule) {
  return seq(rule, repeat(seq(delimiter, rule)))
}

function path(prefix, final) {
  return choice(final, seq(prefix, '.', final))
}

function parenthesize(rule) {
  return seq('(', rule, ')')
}
