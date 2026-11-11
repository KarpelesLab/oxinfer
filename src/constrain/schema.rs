//! JSON, and JSON Schema compiled to a [`Grammar`]: the subset OpenAI's structured outputs
//! take, written the way a constrained decoder needs it (keys in the schema's order, bounded
//! whitespace and numbers).
//!
//! Supported: `type` (a name or a list of names: `object`, `array`, `string`, `number`,
//! `integer`, `boolean`, `null`), `properties` with `required` (properties that are not required
//! may be left out) and `additionalProperties` (`false`, or absent next to `properties`: only
//! the declared keys; `true` or a schema without `properties`: any keys), `items`, `minItems`,
//! `maxItems`, `enum`, `const`, `anyOf`, `$ref` to `#`, `#/$defs/...` or any other place of the
//! document (recursion included), and the annotations (`title`, `description`, `default`,
//! `examples`, `format`, ...), which constrain nothing. Any other keyword is an error: a schema
//! is never partly applied.

use super::grammar::{Grammar, RuleId, Sym};
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// How values are written.
#[derive(Clone, Debug, PartialEq)]
pub enum Dialect {
    Json,
    /// Gemma 4's tool-call arguments: bare keys, strings between `quote` (the `<|"|>` token), no
    /// whitespace.
    Gemma {
        quote: Vec<Sym>,
    },
}

/// Rules shared by every schema compiled into one grammar (strings, numbers, whitespace, ...).
#[derive(Clone, Default)]
pub struct Prims(HashMap<(bool, &'static str), RuleId>);

/// Keywords that constrain nothing.
const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "format",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
];
const KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "$ref",
];
/// The most array items a bound may count (each is a grammar rule).
const MAX_ITEMS: u64 = 10_000;

pub struct Compiler<'a> {
    pub g: &'a mut Grammar,
    prims: &'a mut Prims,
    root: &'a Value,
    dialect: &'a Dialect,
    refs: HashMap<String, RuleId>,
}

/// Compiles `schema` into `g`; the returned rule matches one value of it.
pub fn compile(g: &mut Grammar, prims: &mut Prims, schema: &Value, dialect: &Dialect) -> Result<RuleId> {
    Compiler::new(g, prims, schema, dialect).reference("#", "#")
}

/// Whether `schema` is in the supported subset (the error says what is not).
pub fn check(schema: &Value) -> Result<()> {
    let mut g = Grammar::new();
    compile(&mut g, &mut Prims::default(), schema, &Dialect::Json)?;
    g.validate()
}

/// The rule of any JSON object (`response_format: json_object`).
pub fn json_object(g: &mut Grammar, prims: &mut Prims) -> RuleId {
    let root = Value::Bool(true);
    let mut c = Compiler::new(g, prims, &root, &Dialect::Json);
    let any = c.any();
    c.map_object(any)
}

fn is_type(v: &Value, t: &str) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => false,
    }
}

impl<'a> Compiler<'a> {
    pub fn new(g: &'a mut Grammar, prims: &'a mut Prims, root: &'a Value, dialect: &'a Dialect) -> Self {
        Self {
            g,
            prims,
            root,
            dialect,
            refs: HashMap::new(),
        }
    }

    fn json(&self) -> bool {
        *self.dialect == Dialect::Json
    }

    fn prim(&mut self, name: &'static str, build: impl FnOnce(&mut Self, RuleId)) -> RuleId {
        let key = (self.json(), name);
        if let Some(&r) = self.prims.0.get(&key) {
            return r;
        }
        let r = self.g.new_rule();
        self.prims.0.insert(key, r);
        build(self, r);
        r
    }

    /// Optional whitespace between JSON tokens: nothing, a space, or a newline and up to 16
    /// spaces or tabs (enough for indented output, and never an endless run).
    fn ws(&mut self) -> Vec<Sym> {
        if !self.json() {
            return Vec::new();
        }
        let r = self.prim("ws", |c, r| {
            let (sp, nl, ind) = (c.g.byte(b' '), c.g.byte(b'\n'), c.g.class(&[(b' ', b' '), (b'\t', b'\t')]));
            let mut line = vec![nl];
            line.extend(c.g.repeat(vec![ind], 0, Some(16)));
            c.g.add_alt(r, Vec::new());
            c.g.add_alt(r, vec![sp]);
            c.g.add_alt(r, line);
        });
        vec![Sym::Rule(r)]
    }

    /// One UTF-8 character and then `then`, as alternatives: ASCII from 0x20 without `exclude`
    /// (with newline and tab when `controls`), and every well-formed multi-byte sequence.
    pub fn char_then(&mut self, exclude: &[u8], controls: bool, then: Sym) -> Vec<Vec<Sym>> {
        let g = &mut *self.g;
        let mut ascii: Vec<(u8, u8)> = (0x20..=0x7fu8).filter(|b| !exclude.contains(b)).map(|b| (b, b)).collect();
        if controls {
            ascii.extend([(b'\n', b'\n'), (b'\t', b'\t')]);
        }
        let c = g.class(&[(0x80, 0xbf)]);
        vec![
            vec![g.class(&ascii), then],
            vec![g.class(&[(0xc2, 0xdf)]), c, then],
            vec![g.byte(0xe0), g.class(&[(0xa0, 0xbf)]), c, then],
            vec![g.class(&[(0xe1, 0xec), (0xee, 0xef)]), c, c, then],
            vec![g.byte(0xed), g.class(&[(0x80, 0x9f)]), c, then],
            vec![g.byte(0xf0), g.class(&[(0x90, 0xbf)]), c, c, then],
            vec![g.class(&[(0xf1, 0xf3)]), c, c, c, then],
            vec![g.byte(0xf4), g.class(&[(0x80, 0x8f)]), c, c, then],
        ]
    }

    /// A string of any content.
    fn string(&mut self) -> RuleId {
        self.prim("string", |c, r| {
            let body = c.g.new_rule();
            let again = Sym::Rule(body);
            match c.dialect.clone() {
                Dialect::Json => {
                    let (quote, bs) = (c.g.byte(b'"'), c.g.byte(b'\\'));
                    c.g.add_alt(body, vec![quote]);
                    for a in c.char_then(b"\"\\", false, again) {
                        c.g.add_alt(body, a);
                    }
                    let g = &mut *c.g;
                    let esc = g.class(&b"\"\\/bfnrt".map(|b| (b, b)));
                    g.add_alt(body, vec![bs, esc, again]);
                    // \uXXXX: a code unit outside the surrogates, or a surrogate pair.
                    let (u, h) = (g.byte(b'u'), g.class(&[(b'0', b'9'), (b'a', b'f'), (b'A', b'F')]));
                    let d = g.class(&[(b'd', b'd'), (b'D', b'D')]);
                    let plain = g.class(&[(b'0', b'9'), (b'a', b'c'), (b'A', b'C'), (b'e', b'f'), (b'E', b'F')]);
                    let low = g.class(&[(b'0', b'7')]);
                    let hi = g.class(&[(b'8', b'9'), (b'a', b'b'), (b'A', b'B')]);
                    let lo = g.class(&[(b'c', b'f'), (b'C', b'F')]);
                    g.add_alt(body, vec![bs, u, plain, h, h, h, again]);
                    g.add_alt(body, vec![bs, u, d, low, h, h, again]);
                    g.add_alt(body, vec![bs, u, d, hi, h, h, bs, u, d, lo, h, h, again]);
                    g.add_alt(r, vec![quote, again]);
                }
                Dialect::Gemma { quote } => {
                    c.g.add_alt(body, quote.clone());
                    // The quote written out as text would end the string for the parser.
                    let exclude: &[u8] = if matches!(quote[..], [Sym::Token(_)]) { b"" } else { b"<" };
                    for a in c.char_then(exclude, true, again) {
                        c.g.add_alt(body, a);
                    }
                    let mut s = quote;
                    s.push(again);
                    c.g.add_alt(r, s);
                }
            }
        })
    }

    /// The digits of a number before the fraction: `0`, or up to 15 digits.
    fn int_digits(&mut self) -> Vec<Sym> {
        let r = self.prim("digits", |c, r| {
            let (zero, first, digit) = (c.g.byte(b'0'), c.g.class(&[(b'1', b'9')]), c.g.class(&[(b'0', b'9')]));
            let mut many = vec![first];
            many.extend(c.g.repeat(vec![digit], 0, Some(14)));
            c.g.add_alt(r, vec![zero]);
            c.g.add_alt(r, many);
        });
        let minus = self.g.byte(b'-');
        vec![self.g.opt(vec![minus]), Sym::Rule(r)]
    }

    /// An integer of at most 15 digits (exact in a double); no `-0`, which is not an integer
    /// to every parser.
    fn integer(&mut self) -> RuleId {
        self.prim("integer", |c, r| {
            let (zero, minus, first, digit) = (
                c.g.byte(b'0'),
                c.g.byte(b'-'),
                c.g.class(&[(b'1', b'9')]),
                c.g.class(&[(b'0', b'9')]),
            );
            let mut many = vec![c.g.opt(vec![minus]), first];
            many.extend(c.g.repeat(vec![digit], 0, Some(14)));
            c.g.add_alt(r, vec![zero]);
            c.g.add_alt(r, many);
        })
    }

    /// A number: at most 15 integer digits, 16 fraction digits and 2 exponent digits (every
    /// such number is a finite double).
    fn number(&mut self) -> RuleId {
        self.prim("number", |c, r| {
            let mut n = c.int_digits();
            let g = &mut *c.g;
            let digit = g.class(&[(b'0', b'9')]);
            let mut frac = vec![g.byte(b'.')];
            frac.extend(g.repeat(vec![digit], 1, Some(16)));
            n.push(g.opt(frac));
            let sign = g.class(&[(b'+', b'+'), (b'-', b'-')]);
            let mut exp = vec![g.class(&[(b'e', b'e'), (b'E', b'E')]), g.opt(vec![sign])];
            exp.extend(g.repeat(vec![digit], 1, Some(2)));
            n.push(g.opt(exp));
            g.add_alt(r, n);
        })
    }

    /// Any value.
    pub fn any(&mut self) -> RuleId {
        self.prim("value", |c, r| {
            let (s, n) = (c.string(), c.number());
            let (o, a) = (c.map_object(r), c.array(r, 0, None));
            for x in [o, a, s, n] {
                c.g.add_alt(r, vec![Sym::Rule(x)]);
            }
            for l in ["true", "false", "null"] {
                let l = c.g.lit(l.as_bytes());
                c.g.add_alt(r, l);
            }
        })
    }

    /// An object of any keys with values of rule `value`.
    pub fn map_object(&mut self, value: RuleId) -> RuleId {
        let ws = self.ws();
        let mut member = if self.json() {
            vec![Sym::Rule(self.string())]
        } else {
            // A bare key.
            let first = self.g.class(&[(b'a', b'z'), (b'A', b'Z'), (b'_', b'_')]);
            let more = self.g.class(&[(b'a', b'z'), (b'A', b'Z'), (b'_', b'_'), (b'0', b'9')]);
            let mut k = vec![first];
            k.extend(self.g.repeat(vec![more], 0, None));
            k
        };
        member.push(self.g.byte(b':'));
        member.extend(ws.clone());
        member.push(Sym::Rule(value));
        let member = Sym::Rule(self.g.rule(vec![member]));
        let (open, close, comma) = (self.g.byte(b'{'), self.g.byte(b'}'), self.g.byte(b','));
        let mut next = vec![comma];
        next.extend(ws.clone());
        next.push(member);
        let mut empty = vec![open];
        empty.extend(ws.clone());
        empty.push(close);
        let mut full = vec![open];
        full.extend(ws.clone());
        full.push(member);
        full.extend(self.g.repeat(next, 0, None));
        full.extend(ws);
        full.push(close);
        self.g.rule(vec![empty, full])
    }

    /// An array of `min` to `max` items of rule `item`.
    pub fn array(&mut self, item: RuleId, min: usize, max: Option<usize>) -> RuleId {
        let ws = self.ws();
        let (open, close, comma) = (self.g.byte(b'['), self.g.byte(b']'), self.g.byte(b','));
        let mut alts = Vec::new();
        if min == 0 {
            let mut empty = vec![open];
            empty.extend(ws.clone());
            empty.push(close);
            alts.push(empty);
        }
        if max != Some(0) {
            let mut next = vec![comma];
            next.extend(ws.clone());
            next.push(Sym::Rule(item));
            let mut full = vec![open];
            full.extend(ws.clone());
            full.push(Sym::Rule(item));
            full.extend(self.g.repeat(next, min.saturating_sub(1), max.map(|m| m - 1)));
            full.extend(ws);
            full.push(close);
            alts.push(full);
        }
        self.g.rule(alts)
    }

    /// A value written out (an `enum` or `const` member, a key).
    pub fn literal(&mut self, v: &Value) -> Vec<Sym> {
        match self.dialect.clone() {
            Dialect::Json => self.g.lit(v.to_string().as_bytes()),
            Dialect::Gemma { quote } => match v {
                Value::String(s) => {
                    let mut l = quote.clone();
                    l.extend(self.g.lit(s.as_bytes()));
                    l.extend(quote);
                    l
                }
                Value::Array(a) => {
                    let mut l = self.g.lit(b"[");
                    for (i, x) in a.iter().enumerate() {
                        if i > 0 {
                            l.push(self.g.byte(b','));
                        }
                        l.extend(self.literal(x));
                    }
                    l.push(self.g.byte(b']'));
                    l
                }
                Value::Object(o) => {
                    let mut l = self.g.lit(b"{");
                    for (i, (k, x)) in o.iter().enumerate() {
                        if i > 0 {
                            l.push(self.g.byte(b','));
                        }
                        l.extend(self.g.lit(k.as_bytes()));
                        l.push(self.g.byte(b':'));
                        l.extend(self.literal(x));
                    }
                    l.push(self.g.byte(b'}'));
                    l
                }
                other => self.g.lit(other.to_string().as_bytes()),
            },
        }
    }

    /// The schema a `$ref` chain ends at.
    pub fn resolve(&self, mut s: &'a Value, path: &str) -> Result<&'a Value> {
        for _ in 0..32 {
            let Some(r) = s.get("$ref") else { return Ok(s) };
            s = self.target(r, path)?;
        }
        bail!("{path}: $ref chain too long")
    }

    fn target(&self, r: &Value, path: &str) -> Result<&'a Value> {
        let ptr = r.as_str().with_context(|| format!("{path}: $ref must be a string"))?;
        let Some(rest) = ptr.strip_prefix('#') else {
            bail!("{path}: $ref {ptr:?} is not supported (only references inside the schema, #/...)");
        };
        self.root.pointer(rest).with_context(|| format!("{path}: $ref {ptr:?} not found"))
    }

    /// The rule of the schema at pointer `ptr` (compiled once: recursion refers back to it).
    fn reference(&mut self, ptr: &str, path: &str) -> Result<RuleId> {
        if let Some(&r) = self.refs.get(ptr) {
            return Ok(r);
        }
        let target = self.target(&Value::String(ptr.to_string()), path)?;
        let r = self.g.new_rule();
        self.refs.insert(ptr.to_string(), r);
        let t = self.node(target, ptr)?;
        self.g.add_alt(r, vec![Sym::Rule(t)]);
        Ok(r)
    }

    /// Checks the keywords of a schema object and returns it; None for the schema `true`.
    fn keywords<'v>(&self, s: &'v Value, path: &str) -> Result<Option<&'v Map<String, Value>>> {
        let o = match s {
            Value::Bool(true) => return Ok(None),
            Value::Bool(false) => bail!("{path}: the schema `false` matches nothing"),
            Value::Object(o) => o,
            _ => bail!("{path}: a schema must be an object or a boolean"),
        };
        if let Some(k) = o
            .keys()
            .find(|k| !KEYWORDS.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()))
        {
            bail!("{path}: schema keyword {k:?} is not supported");
        }
        for single in ["$ref", "anyOf"] {
            if o.contains_key(single)
                && let Some(k) = o.keys().find(|k| *k != single && KEYWORDS.contains(&k.as_str()))
            {
                bail!("{path}: {k:?} next to {single:?} is not supported");
            }
        }
        Ok(Some(o))
    }

    /// The types a schema object allows (None: any).
    fn types<'v>(o: &'v Map<String, Value>, path: &str) -> Result<Option<Vec<&'v str>>> {
        let known = |t: &'v str| -> Result<&'v str> {
            if !["object", "array", "string", "number", "integer", "boolean", "null"].contains(&t) {
                bail!("{path}: unknown type {t:?}");
            }
            Ok(t)
        };
        Ok(match o.get("type") {
            Some(Value::String(t)) => Some(vec![known(t)?]),
            Some(Value::Array(a)) if !a.is_empty() => Some(
                a.iter()
                    .map(|t| known(t.as_str().with_context(|| format!("{path}: type names must be strings"))?))
                    .collect::<Result<_>>()?,
            ),
            Some(_) => bail!("{path}: type must be a name or a list of names"),
            // Keywords of one type imply it.
            None if ["properties", "required", "additionalProperties"]
                .iter()
                .any(|k| o.contains_key(*k)) =>
            {
                Some(vec!["object"])
            }
            None if ["items", "minItems", "maxItems"].iter().any(|k| o.contains_key(*k)) => Some(vec!["array"]),
            None => None,
        })
    }

    /// The members of `enum` (of the schema's types) or `const`.
    fn choices<'v>(o: &'v Map<String, Value>, path: &str) -> Result<Option<Vec<&'v Value>>> {
        if let Some(c) = o.get("const") {
            return Ok(Some(vec![c]));
        }
        let Some(e) = o.get("enum") else { return Ok(None) };
        let e = e.as_array().with_context(|| format!("{path}: enum must be an array"))?;
        let types = Self::types(o, path)?;
        let vals: Vec<&Value> = e
            .iter()
            .filter(|v| types.as_ref().is_none_or(|ts| ts.iter().any(|t| is_type(v, t))))
            .collect();
        if vals.is_empty() {
            bail!("{path}: enum has no value of the schema's type");
        }
        Ok(Some(vals))
    }

    /// The rule of one value of schema `s`.
    pub fn node(&mut self, s: &Value, path: &str) -> Result<RuleId> {
        let Some(o) = self.keywords(s, path)? else { return Ok(self.any()) };
        if let Some(r) = o.get("$ref") {
            let ptr = r.as_str().with_context(|| format!("{path}: $ref must be a string"))?;
            return self.reference(ptr, path);
        }
        if let Some(vals) = Self::choices(o, path)? {
            let alts = vals.into_iter().map(|v| self.literal(v)).collect();
            return Ok(self.g.rule(alts));
        }
        if let Some(a) = o.get("anyOf") {
            let a = a
                .as_array()
                .filter(|a| !a.is_empty())
                .with_context(|| format!("{path}: anyOf must be a non-empty array"))?;
            let mut alts = Vec::new();
            for (i, s) in a.iter().enumerate() {
                alts.push(vec![Sym::Rule(self.node(s, &format!("{path}/anyOf/{i}"))?)]);
            }
            return Ok(self.g.rule(alts));
        }
        let Some(types) = Self::types(o, path)? else {
            return Ok(self.any());
        };
        let mut alts = Vec::new();
        for t in types {
            alts.push(match t {
                "object" => vec![Sym::Rule(self.object(o, path)?)],
                "array" => vec![Sym::Rule(self.array_of(o, path)?)],
                "string" => vec![Sym::Rule(self.string())],
                "number" => vec![Sym::Rule(self.number())],
                "integer" => vec![Sym::Rule(self.integer())],
                "boolean" => {
                    let (t, f) = (self.g.lit(b"true"), self.g.lit(b"false"));
                    vec![Sym::Rule(self.g.rule(vec![t, f]))]
                }
                _ => self.g.lit(b"null"),
            });
        }
        Ok(self.g.rule(alts))
    }

    fn array_of(&mut self, o: &Map<String, Value>, path: &str) -> Result<RuleId> {
        let count = |k: &str| -> Result<Option<usize>> {
            match o.get(k) {
                None => Ok(None),
                Some(v) => match v.as_u64() {
                    Some(n) if n <= MAX_ITEMS => Ok(Some(n as usize)),
                    Some(_) => bail!("{path}: {k} above {MAX_ITEMS} is not supported"),
                    None => bail!("{path}: {k} must be a non-negative integer"),
                },
            }
        };
        let (min, max) = (count("minItems")?.unwrap_or(0), count("maxItems")?);
        if max.is_some_and(|m| m < min) {
            bail!("{path}: maxItems is below minItems");
        }
        let item = match o.get("items") {
            Some(s) => self.node(s, &format!("{path}/items"))?,
            None => self.any(),
        };
        Ok(self.array(item, min, max))
    }

    /// The declared properties of an object schema, in order, with whether each is required;
    /// None when the object takes any keys (then with the schema of their values).
    #[allow(clippy::type_complexity)]
    pub fn properties<'v>(o: &'v Map<String, Value>, path: &str) -> Result<Result<Vec<(&'v String, &'v Value, bool)>, Option<&'v Value>>> {
        let props = match o.get("properties") {
            None => None,
            Some(Value::Object(p)) => Some(p),
            Some(_) => bail!("{path}: properties must be an object"),
        };
        let required: Vec<&str> = match o.get("required") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().with_context(|| format!("{path}: required must be an array of names")))
                .collect::<Result<_>>()?,
            Some(_) => bail!("{path}: required must be an array of names"),
        };
        if let Some(r) = required.iter().find(|r| !props.is_some_and(|p| p.contains_key(**r))) {
            bail!("{path}: required property {r:?} is not in properties");
        }
        let open = match o.get("additionalProperties") {
            None | Some(Value::Bool(false)) => None,
            Some(s) => Some(s),
        };
        match (props.filter(|p| !p.is_empty()), open) {
            (Some(_), Some(_)) => bail!("{path}: additionalProperties next to properties is not supported (set it to false)"),
            (Some(p), None) => Ok(Ok(p.iter().map(|(k, s)| (k, s, required.contains(&k.as_str()))).collect())),
            // `{"type": "object"}` alone takes anything; with `properties: {}` or
            // `additionalProperties: false` it is the empty object.
            (None, None) if props.is_none() && !o.contains_key("additionalProperties") => Ok(Err(None)),
            (None, None) => Ok(Ok(Vec::new())),
            (None, Some(s)) => Ok(Err(Some(s))),
        }
    }

    /// The object schema that `s` resolves to (tool parameters); an error when it is not one.
    pub fn object_schema(&self, s: &'a Value, path: &str) -> Result<&'a Map<String, Value>> {
        let s = self.resolve(s, path)?;
        let o = self.keywords(s, path)?;
        let o = o.with_context(|| format!("{path}: expected an object schema"))?;
        let object = match Self::types(o, path)? {
            Some(t) => t == ["object"],
            None => !["enum", "const", "anyOf"].iter().any(|k| o.contains_key(*k)),
        };
        if !object {
            bail!("{path}: expected an object schema");
        }
        Ok(o)
    }

    /// The rule of the object schema `s` (see [`Compiler::object_schema`]).
    pub fn object_value(&mut self, s: &'a Value, path: &str) -> Result<RuleId> {
        let o = self.object_schema(s, path)?;
        self.object(o, path)
    }

    fn object(&mut self, o: &Map<String, Value>, path: &str) -> Result<RuleId> {
        let props = match Self::properties(o, path)? {
            // Gemma's template writes the arguments of a call sorted by key.
            Ok(mut p) => {
                if !self.json() {
                    p.sort_by(|a, b| a.0.cmp(b.0));
                }
                p
            }
            Err(values) => {
                let v = match values {
                    Some(s) => self.node(s, &format!("{path}/additionalProperties"))?,
                    None => self.any(),
                };
                return Ok(self.map_object(v));
            }
        };
        let mut members = Vec::new();
        for (k, s, required) in props {
            let mut m = if self.json() {
                self.g.lit(Value::String(k.clone()).to_string().as_bytes())
            } else {
                self.g.lit(k.as_bytes())
            };
            m.push(self.g.byte(b':'));
            m.extend(self.ws());
            m.push(Sym::Rule(self.node(s, &format!("{path}/properties/{k}"))?));
            members.push((m, required));
        }
        // tail[first]: what follows from property i on, by whether a member was written yet.
        let ws = self.ws();
        let (open, close, comma) = (self.g.byte(b'{'), self.g.byte(b'}'), self.g.byte(b','));
        let mut end = ws.clone();
        end.push(close);
        let end = self.g.rule(vec![end]);
        let mut tail = [end, end];
        for (m, required) in members.into_iter().rev() {
            let mut next = tail;
            for (first, slot) in next.iter_mut().enumerate() {
                let mut with = if first == 1 { Vec::new() } else { vec![comma] };
                with.extend(ws.clone());
                with.extend(m.clone());
                with.push(Sym::Rule(tail[0]));
                let mut alts = vec![with];
                if !required {
                    alts.push(vec![Sym::Rule(tail[first])]);
                }
                *slot = self.g.rule(alts);
            }
            tail = next;
        }
        Ok(self.g.rule(vec![vec![open, Sym::Rule(tail[1])]]))
    }

    /// For tool-call syntaxes that write strings bare and other values as JSON: the
    /// alternatives of schema `s` as (bare strings, JSON values). `bare` matches any bare string.
    #[allow(clippy::type_complexity)]
    pub fn split(&mut self, s: &'a Value, path: &str, bare: RuleId, depth: usize) -> Result<(Vec<Vec<Sym>>, Vec<Vec<Sym>>)> {
        if depth > 32 {
            bail!("{path}: the schema refers to itself without a value in between");
        }
        let s = self.resolve(s, path)?;
        let Some(o) = self.keywords(s, path)? else {
            return Ok((vec![vec![Sym::Rule(bare)]], vec![vec![Sym::Rule(self.any())]]));
        };
        let (mut text, mut json) = (Vec::new(), Vec::new());
        if let Some(vals) = Self::choices(o, path)? {
            for v in vals {
                match v {
                    Value::String(t) => text.push(self.g.lit(t.as_bytes())),
                    other => json.push(self.literal(other)),
                }
            }
        } else if let Some(Value::Array(a)) = o.get("anyOf") {
            for (i, b) in a.iter().enumerate() {
                let (t, j) = self.split(b, &format!("{path}/anyOf/{i}"), bare, depth + 1)?;
                text.extend(t);
                json.extend(j);
            }
        } else {
            match Self::types(o, path)? {
                Some(types) if types.contains(&"string") => {
                    text.push(vec![Sym::Rule(bare)]);
                    let rest: Vec<Value> = types.iter().filter(|t| **t != "string").map(|t| (*t).into()).collect();
                    if !rest.is_empty() {
                        let mut other = o.clone();
                        other.insert("type".into(), Value::Array(rest));
                        json.push(vec![Sym::Rule(self.node(&Value::Object(other), path)?)]);
                    }
                }
                Some(_) => json.push(vec![Sym::Rule(self.node(s, path)?)]),
                None => {
                    text.push(vec![Sym::Rule(bare)]);
                    json.push(vec![Sym::Rule(self.any())]);
                }
            }
        }
        Ok((text, json))
    }
}

#[cfg(test)]
mod tests {
    use super::super::grammar::Matcher;
    use super::*;
    use serde_json::json;

    fn grammar(schema: &Value) -> (Grammar, RuleId) {
        let mut g = Grammar::new();
        let r = compile(&mut g, &mut Prims::default(), schema, &Dialect::Json).unwrap();
        g.validate().unwrap();
        (g, r)
    }

    /// Every `ok` is a whole match (and every prefix of it a prefix), every `bad` is not.
    fn check_sets(schema: Value, ok: &[&str], bad: &[&str]) {
        let (g, r) = grammar(&schema);
        for t in ok {
            assert_eq!(Matcher::matches(&g, r, t.as_bytes()), (true, true), "{schema} should accept {t}");
            for n in 0..t.len() {
                assert!(Matcher::matches(&g, r, &t.as_bytes()[..n]).1, "{schema}: prefix {n} of {t}");
            }
            serde_json::from_str::<Value>(t).unwrap();
        }
        for t in bad {
            assert!(!Matcher::matches(&g, r, t.as_bytes()).0, "{schema} should reject {t}");
        }
    }

    #[test]
    fn any_json_object() {
        let mut g = Grammar::new();
        let r = json_object(&mut g, &mut Prims::default());
        g.validate().unwrap();
        let ok = [
            "{}",
            r#"{"a":1}"#,
            r#"{"a": [1, 2.5e-3, true, null, "x"], "b": {"c": {}}}"#,
            "{\n  \"a\": \"\\u00e9\\n\\\"\",\n  \"b\": -0.5\n}",
            r#"{"é": "日本語 😀", "pair": "😀"}"#,
        ];
        for t in ok {
            assert_eq!(Matcher::matches(&g, r, t.as_bytes()), (true, true), "{t}");
            serde_json::from_str::<Value>(t).unwrap();
        }
        let bad = [
            "[]",
            "1",
            r#"{"a":01}"#,
            r#"{"a":1,}"#,
            r#"{a:1}"#,
            r#"{"a":1}}"#,
            r#"{"a":1} "#,
            "{\"a\":\"\n\"}",
            r#"{"a":"\x"}"#,
            r#"{"a":"\ud83d"}"#,
            r#"{"a":"\ude00"}"#,
            r#"{"a":1e999}"#,
            "{\"a\":   1}",
            "{\"a\":\"\u{0}\"}",
        ];
        for t in bad {
            assert!(!Matcher::matches(&g, r, t.as_bytes()).0, "{t}");
        }
        // Bytes that are not UTF-8 never start a match.
        for t in [&b"{\"\xff"[..], b"{\"\xc0\xaf", b"{\"\xed\xa0\x80", b"{\"\xe3\x81\""] {
            assert_eq!(Matcher::matches(&g, r, t), (false, false), "{t:?}");
        }
        assert_eq!(Matcher::matches(&g, r, b"{\"\xe3\x81"), (false, true));
    }

    #[test]
    fn objects_follow_the_schema() {
        let person = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "age", "tags"],
            "additionalProperties": false,
        });
        check_sets(
            person,
            &[
                r#"{"name":"Ada","age":36,"tags":[]}"#,
                r#"{"name": "Ada", "age": -1, "tags": ["a", "b"]}"#,
                "{\n\t\"name\": \"\",\n\t\"age\": 0,\n\t\"tags\": [\n\t\t\"x\"\n\t]\n}",
            ],
            &[
                r#"{"name":"Ada","age":36}"#,
                r#"{"age":36,"name":"Ada","tags":[]}"#,
                r#"{"name":"Ada","age":36.5,"tags":[]}"#,
                r#"{"name":"Ada","age":36,"tags":[1]}"#,
                r#"{"name":"Ada","age":36,"tags":[],"x":1}"#,
                r#"{"name":1,"age":36,"tags":[]}"#,
                r#"{}"#,
            ],
        );
        // Properties that are not required may be left out, in any combination.
        let optional = json!({
            "type": "object",
            "properties": {"a": {"type": "boolean"}, "b": {"type": "null"}, "c": {"type": "number"}},
            "required": ["b"],
        });
        check_sets(
            optional,
            &[
                r#"{"b":null}"#,
                r#"{"a":true,"b":null}"#,
                r#"{"b":null,"c":1.5}"#,
                r#"{"a":false,"b":null,"c":2e3}"#,
            ],
            &[
                r#"{}"#,
                r#"{"a":true}"#,
                r#"{"a":true,"c":1}"#,
                r#"{,"b":null}"#,
                r#"{"b":null,}"#,
                r#"{"c":1,"b":null}"#,
            ],
        );
        check_sets(
            json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}),
            &["{}", r#"{"a":1}"#, r#"{"b":1}"#, r#"{"a":1,"b":2}"#],
            &[r#"{"b":1,"a":2}"#, r#"{"c":1}"#],
        );
        // No declared properties: the empty object, or any object.
        check_sets(json!({"type": "object", "properties": {}}), &["{}", "{ }"], &[r#"{"a":1}"#]);
        check_sets(json!({"type": "object"}), &["{}", r#"{"a":[1]}"#], &["[]"]);
        check_sets(
            json!({"type": "object", "additionalProperties": {"type": "integer"}}),
            &["{}", r#"{"a":1,"b":2}"#],
            &[r#"{"a":"x"}"#],
        );
    }

    #[test]
    fn scalars_enums_and_unions() {
        check_sets(
            json!({"type": "string"}),
            &[r#""""#, r#""a\tb""#, "\"é\""],
            &["a", r#""a"#, r#""a"b""#],
        );
        check_sets(
            json!({"type": "integer"}),
            &["0", "-7", "123456789012345"],
            &["1.0", "01", "+1", "1e3", "1234567890123456", "-", "-0"],
        );
        check_sets(
            json!({"type": "number"}),
            &["0", "-0.5", "3.14", "1e10", "2.5E-3", "-1e+5"],
            &["1.", ".5", "1e", "1e100", "0x1", "NaN"],
        );
        check_sets(json!({"type": "boolean"}), &["true", "false"], &["True", "1", "null"]);
        check_sets(json!({"type": ["string", "null"]}), &["null", r#""x""#], &["1", "true"]);
        check_sets(
            json!({"enum": ["red", "green", 3, null, {"a": [1]}]}),
            &[r#""red""#, r#""green""#, "3", "null", r#"{"a":[1]}"#],
            &[r#""blue""#, "4", r#""re""#],
        );
        check_sets(json!({"type": "string", "enum": ["a", 1, "b"]}), &[r#""a""#, r#""b""#], &["1"]);
        check_sets(json!({"const": "only \"this\""}), &[r#""only \"this\"""#], &[r#""only""#]);
        check_sets(
            json!({"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"k": {"const": 1}}, "required": ["k"]}]}),
            &["5", r#"{"k":1}"#],
            &[r#"{"k":2}"#, "5.5", "{}"],
        );
        check_sets(json!(true), &["1", r#"{"a":[]}"#, "null"], &["{", "nul"]);
        check_sets(json!({"description": "anything", "format": "x"}), &["[1]", r#""s""#], &["]"]);
    }

    #[test]
    fn arrays_and_bounds() {
        check_sets(
            json!({"type": "array", "items": {"type": "integer"}}),
            &["[]", "[1]", "[1, 2,3]"],
            &["[1,]", "[,1]", "[\"a\"]"],
        );
        check_sets(
            json!({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}),
            &["[1,2]", "[1,2,3]"],
            &["[]", "[1]", "[1,2,3,4]"],
        );
        check_sets(json!({"type": "array", "maxItems": 0}), &["[]"], &["[1]"]);
        check_sets(
            json!({"type": "array", "minItems": 1, "maxItems": 1}),
            &["[null]"],
            &["[]", "[1,2]"],
        );
        check_sets(json!({"items": {"type": "boolean"}}), &["[true,false]"], &["[1]", "{}"]);
    }

    #[test]
    fn references_and_recursion() {
        let tree = json!({
            "type": "object",
            "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#"}}},
            "required": ["value", "children"],
            "additionalProperties": false,
        });
        check_sets(
            tree,
            &[
                r#"{"value":1,"children":[]}"#,
                r#"{"value":1,"children":[{"value":2,"children":[{"value":3,"children":[]}]},{"value":4,"children":[]}]}"#,
            ],
            &[r#"{"value":1,"children":[{"value":2}]}"#, r#"{"value":1,"children":[1]}"#],
        );
        let list = json!({
            "$defs": {
                "node": {"type": "object", "properties": {"v": {"$ref": "#/$defs/num"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
                         "required": ["v", "next"], "additionalProperties": false},
                "num": {"type": "number", "description": "a number"},
            },
            "$ref": "#/$defs/node",
        });
        check_sets(
            list,
            &[r#"{"v":1,"next":null}"#, r#"{"v":1,"next":{"v":2.5,"next":{"v":3,"next":null}}}"#],
            &[r#"{"v":1,"next":{}}"#, r#"{"v":"1","next":null}"#],
        );
        check_sets(
            json!({"definitions": {"a/b": {"const": 1}}, "type": "array", "items": {"$ref": "#/definitions/a~1b"}}),
            &["[1,1]"],
            &["[2]"],
        );
    }

    #[test]
    fn unsupported_schemas_are_errors() {
        let err = |s: Value| check(&s).unwrap_err().to_string();
        assert!(err(json!({"type": "string", "pattern": "^a"})).contains("\"pattern\" is not supported"));
        assert!(
            err(json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}}))
                .contains("#/properties/n: schema keyword \"minimum\"")
        );
        for s in [
            json!({"oneOf": [{"type": "string"}]}),
            json!({"allOf": [{"type": "string"}]}),
            json!({"not": {}}),
            json!({"type": "string", "maxLength": 3}),
            json!({"type": "array", "prefixItems": []}),
            json!({"type": "object", "patternProperties": {}}),
            json!({"type": "widget"}),
            json!({"type": 3}),
            json!(false),
            json!("string"),
            json!({"$ref": "https://example.com/schema.json"}),
            json!({"$ref": "#/$defs/missing"}),
            json!({"$ref": "#", "type": "object"}),
            json!({"type": "object", "properties": {"a": {}}, "additionalProperties": true}),
            json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
            json!({"type": "array", "minItems": 3, "maxItems": 2}),
            json!({"type": "array", "maxItems": 1_000_000}),
            json!({"type": "integer", "enum": ["a"]}),
            json!({"anyOf": []}),
        ] {
            assert!(check(&s).is_err(), "{s}");
        }
        // References that never reach a value.
        assert!(err(json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"})).contains("left-recursive"));
        assert!(err(json!({"anyOf": [{"$ref": "#"}, {"type": "null"}]})).contains("left-recursive"));
        // Annotations and containers are fine.
        check(&json!({"$schema": "x", "title": "t", "type": "string", "format": "date-time", "default": "", "examples": []})).unwrap();
    }

    #[test]
    fn gemma_arguments() {
        let mut g = Grammar::new();
        let d = Dialect::Gemma {
            quote: vec![Sym::Token(52)],
        };
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
            "unit": {"enum": ["c", "f"]}}, "required": ["city"]});
        let r = compile(&mut g, &mut Prims::default(), &schema, &d).unwrap();
        g.validate().unwrap();
        // With the quote token as Q: {city:QParisQ,days:3,unit:QcQ}
        let mut m = Matcher::new();
        let mut state = m.start(&g, r, super::super::grammar::DONE);
        let feed = |m: &mut Matcher, state: &mut Vec<u32>, text: &str| {
            for part in text.split_inclusive('Q') {
                let (bytes, q) = match part.strip_suffix('Q') {
                    Some(b) => (b, true),
                    None => (part, false),
                };
                for &b in bytes.as_bytes() {
                    let mut next = Vec::new();
                    m.step_byte(&g, state, b, &mut next);
                    *state = next;
                }
                if q {
                    let mut next = Vec::new();
                    m.step_token(&g, state, 52, &mut next);
                    *state = next;
                }
            }
        };
        feed(&mut m, &mut state, "{city:QParis, \"France\"Q,days:3,unit:QcQ}");
        assert!(state.contains(&super::super::grammar::DONE));
        let mut state = m.start(&g, r, super::super::grammar::DONE);
        feed(&mut m, &mut state, "{city:QxQ}");
        assert!(state.contains(&super::super::grammar::DONE));
        let mut state = m.start(&g, r, super::super::grammar::DONE);
        feed(&mut m, &mut state, "{city:QxQ, ");
        assert!(state.is_empty());
        let mut state = m.start(&g, r, super::super::grammar::DONE);
        feed(&mut m, &mut state, "{\"city\"");
        assert!(state.is_empty());
    }
}
