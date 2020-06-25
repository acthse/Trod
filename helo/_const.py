from . import util

OPERATOR = util.adict(
    AND='AND',
    OR='OR',
    ADD='+',
    SUB='-',
    MUL='*',
    DIV='/',
    BIN_AND='&',
    BIN_OR='|',
    XOR='#',
    MOD='%',
    EQ='=',
    LT='<',
    LTE='<=',
    GT='>',
    GTE='>=',
    NE='!=',
    IN='IN',
    NOT_IN='NOT IN',
    IS='IS',
    IS_NOT='IS NOT',
    LIKE='LIKE BINARY',
    ILIKE='LIKE',
    EXISTS='EXISTS',
    NEXISTS='NOT EXISTS',
    BETWEEN='BETWEEN',
    NBETWEEN='NOT BETWEEN',
    REGEXP='REGEXP BINARY',
    IREGEXP='REGEXP',
    BITWISE_NEGATION='~',
    CONCAT='||',
)
MYSQL_ENGINE = util.adict(
    innodb="InnoDB",
    myisam="MyISAM",
)
ENCODINGS = (
    'ig5',
    'ec8',
    'p850',
    'p8',
    'oi8r',
    'atin1',
    'atin2',
    'ascii',
    'ujis',
    'sjis',
    'hebrew',
    'tis620',
    'euckr',
    'gb2312',
    'macce',
    'cp1251',
    'macroman',
    'cp1257',
    'binary',
    'armscii8',
    'cp1256',
    'cp866',
    'dec8',
    'greek',
    'hp8',
    'keybcs2',
    'koi8r',
    'koi8u',
    'latin2',
    'latin5',
    'latin7',
    'cp850',
    'cp852',
    'swe7',
    'big5',
    'gbk',
    'geostd8',
    'latin1',
    'cp932',
    'eucjpms',
    'cp1250',
    'utf16',
    'ucs2',
    'utf32',
    'utf8',
    'utf8mb4',
)
