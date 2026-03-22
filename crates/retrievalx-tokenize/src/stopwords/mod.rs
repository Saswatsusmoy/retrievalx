#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;

static STOPWORDS: Lazy<HashMap<&'static str, HashSet<&'static str>>> = Lazy::new(|| {
    let mut map = HashMap::new();

    map.insert(
        "en",
        set(&["the", "is", "at", "which", "on", "a", "an", "and", "of"]),
    );
    map.insert(
        "fr",
        set(&["le", "la", "les", "de", "des", "et", "un", "une"]),
    );
    map.insert(
        "de",
        set(&["der", "die", "das", "und", "ein", "eine", "mit"]),
    );
    map.insert(
        "es",
        set(&["el", "la", "los", "las", "de", "y", "un", "una"]),
    );
    map.insert("pt", set(&["o", "a", "os", "as", "de", "e", "um", "uma"]));
    map.insert("it", set(&["il", "lo", "la", "gli", "le", "di", "e", "un"]));
    map.insert("nl", set(&["de", "het", "een", "en", "van", "op", "in"]));
    map.insert("ru", set(&["и", "в", "во", "не", "что", "он", "на"]));
    map.insert("ar", set(&["و", "في", "من", "على", "أن", "إلى"]));
    map.insert("hi", set(&["और", "का", "के", "में", "से", "पर", "यह"]));
    map.insert("zh", set(&["的", "了", "在", "是", "和", "有"]));
    map.insert("ja", set(&["の", "に", "は", "を", "た", "が"]));
    map.insert("ko", set(&["의", "에", "이", "가", "은", "는"]));
    map.insert("sv", set(&["och", "det", "att", "i", "en", "som"]));
    map.insert("da", set(&["og", "i", "jeg", "det", "at", "en"]));
    map.insert("no", set(&["og", "i", "jeg", "det", "at", "en"]));
    map.insert("fi", set(&["ja", "on", "se", "että", "ei", "kun"]));
    map.insert("ro", set(&["și", "în", "la", "de", "un", "o"]));
    map.insert("tr", set(&["ve", "bir", "bu", "için", "ile", "de"]));
    map.insert("pl", set(&["i", "w", "na", "to", "z", "że"]));
    map.insert("cs", set(&["a", "v", "je", "na", "se", "to"]));

    map
});

pub fn language_stopwords(lang: &str) -> HashSet<&'static str> {
    STOPWORDS.get(lang).cloned().unwrap_or_default()
}

pub fn supported_languages() -> HashSet<&'static str> {
    STOPWORDS.keys().copied().collect()
}

fn set(words: &[&'static str]) -> HashSet<&'static str> {
    words.iter().copied().collect()
}
