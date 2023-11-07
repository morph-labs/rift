const

min1 = (one, two) => choice(seq(one, optional(two)), seq(optional(one), two))
sep0 = (rule, separator) => optional(sep1(rule, separator))
sep1 = (rule, separator) => seq(rule, repeat(seq(separator, rule)))
sep2 = (rule, separator) => seq(rule, repeat1(seq(separator, rule)))
sep1_ = (rule, separator) => seq(sep1(rule, separator), optional(separator))

class Parser {
  constructor(rules_fn) {
    this.rules_fn = rules_fn
  }

  all($) {
    return choice(...this.rules_fn($));
  }

  forbid($, ...disallow) {
    var seen = new Set(disallow);
    var rules = this.rules_fn($)
    var filtered = rules.filter(rule => !seen.delete(rule.name));
    if (seen.size > 0) {
      throw `Disallowed rules that weren't present:\n\
       ${disallow.toString()}\n\
       ${rules.map(rule => rule.name).toString()}`;
    }
    return choice(...filtered);
  }
}

module.exports = {
  Parser,
  min1,
  sep0,
  sep1,
  sep2,
  sep1_,
}
