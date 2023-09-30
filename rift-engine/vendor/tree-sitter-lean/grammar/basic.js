const PREC = {
  max: 1024,
  arg: 1023,
  lead: 1022,
  min: 10,
}
module.exports = {
  PREC,

  rules: {
    quoted_char: $ => token(
      seq(
        '\\', choice(
          /u[a-fA-F\d]{4}/,
          /x[a-fA-F\d]{2}/,
          /[\\"'rnt]/,
        ),
      ),
    ),
  }
}
