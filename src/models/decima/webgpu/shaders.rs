use super::super::D;

const PARAMS: &str = "struct Params { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32> }";
const GELU: &str = r#"
fn gelu(v: vec4<f32>) -> vec4<f32> {
    let a = abs(v * 0.7071067811865475);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = (((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;
    let erf = sign(v) * (1.0 - poly * exp(-a*a));
    return 0.5*v*(1.0+erf);
}
"#;

// Sum over a 128-thread workgroup, in the same order on every row.
const TOTAL: &str = r#"var<workgroup> red: array<f32, 128>;
fn total(v: f32, t: u32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] += red[t + s]; }
        workgroupBarrier();
    }
    let result = red[0];
    workgroupBarrier();
    return result;
}"#;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Epilogue {
    Bias,
    Gelu,
    Residual,
}
// Long products (k > SLICE) are summed per SLICE-wide k slice and the slice sums added in order,
// so a split-K pass with an ordered reduce gives the same bits as one pass.
pub(super) const SLICE: usize = 384;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Accumulate {
    Whole,
    Sliced,
    // One slice per workgroup z, written as partials for `reduce`.
    Split,
}

// y = x·w + bias for x [m×k] and w stored transposed as [k×n], with n a multiple of 64 and k
// of 16. A workgroup computes `rows` × 64 outputs and each thread `per_thread` rows × 4 columns.
// The 16-step k tile is unrolled into named accumulators, which WebKit keeps in registers.
// Every shape accumulates k in increasing order with fma, so results do not depend on the tile
// or on which rows share a batch.
pub(super) fn gemm(
    rows: usize,
    per_thread: usize,
    epilogue: Epilogue,
    accumulate: Accumulate,
) -> String {
    let bindings = match (accumulate, epilogue) {
        (Accumulate::Split, _) => {
            "@group(0) @binding(2) var<storage, read_write> part: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;"
        }
        (_, Epilogue::Residual) => {
            "@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> r: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p: Params;"
        }
        _ => {
            "@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: Params;"
        }
    };
    let store = match (accumulate, epilogue) {
        (Accumulate::Split, _) => "part[(g.z * m + row) * n4 + col] = v",
        (_, Epilogue::Bias) => "y[row * n4 + col] = v + bias[col]",
        (_, Epilogue::Gelu) => "y[row * n4 + col] = gelu(v + bias[col])",
        (_, Epilogue::Residual) => "y[row * n4 + col] = r[row * n4 + col] + (v + bias[col])",
    };
    let ty = rows / per_thread;
    let threads = 16 * ty;
    assert!(ty.is_power_of_two() && threads <= 256);
    let each = |f: &dyn Fn(usize) -> String| (0..per_thread).map(f).collect::<String>();
    let zero = each(&|i| format!("var c{i} = vec4<f32>(0.0); "));
    let mut steps = String::new();
    for kk in 0..16 {
        steps += "        {\n";
        steps += &each(&|i| format!("            let a{i} = at[{kk}u * BM + ty * TM + {i}u];\n"));
        steps += &format!("            let b = bt[{kk}u * 16u + tx];\n");
        steps += &each(&|i| format!("            c{i} = fma(vec4<f32>(a{i}), b, c{i});\n"));
        steps += "        }\n";
    }
    let tile = format!(
        r#"for (var k0 = start; k0 < end; k0 += 16u) {{
        for (var i = li; i < BM * 4u; i += {threads}u) {{
            let ar = i / 4u; let ak = (i % 4u) * 4u;
            var a = vec4<f32>(0.0);
            if (row0 + ar < m) {{ a = x[(row0 + ar) * k4 + (k0 + ak) / 4u]; }}
            // Scalar stores: threads writing lanes of one shared vec4 would race.
            at[(ak + 0u) * BM + ar] = a.x;
            at[(ak + 1u) * BM + ar] = a.y;
            at[(ak + 2u) * BM + ar] = a.z;
            at[(ak + 3u) * BM + ar] = a.w;
        }}
        for (var i = li; i < 256u; i += {threads}u) {{
            bt[i] = w[(k0 + i / 16u) * n4 + col0 + i % 16u];
        }}
        workgroupBarrier();
{steps}        workgroupBarrier();
    }}"#
    );
    let (body, result) = match accumulate {
        Accumulate::Whole => (
            format!("{zero}\n    let start = 0u; let end = k;\n    {tile}"),
            "c",
        ),
        Accumulate::Split => (
            format!(
                "{zero}\n    let start = g.z * {SLICE}u; let end = start + {SLICE}u;\n    {tile}"
            ),
            "c",
        ),
        Accumulate::Sliced => (
            format!(
                "{totals}\n    for (var start = 0u; start < k; start += {SLICE}u) {{\n    let end = start + {SLICE}u;\n    {zero}\n    {tile}\n    {add}\n    }}",
                totals = each(&|i| format!("var t{i} = vec4<f32>(0.0); ")),
                add = each(&|i| format!("t{i} = t{i} + c{i}; ")),
            ),
            "t",
        ),
    };
    let stores = each(&|i| {
        format!(
            "    {{ let row = row0 + ty * TM + {i}u; let v = {result}{i}; if (row < m) {{ {store}; }} }}\n"
        )
    });
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
{bindings}
{GELU}
const BM = {rows}u; const TM = {per_thread}u;
var<workgroup> at: array<f32, {at}>;
var<workgroup> bt: array<vec4<f32>, 256>;
@compute @workgroup_size(16, {ty})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let m = p.a.x; let n4 = p.a.y / 4u; let k = p.a.z; let k4 = k / 4u;
    // Masked ids let the compiler drop bounds checks on the workgroup arrays.
    let tx = l.x & 15u; let ty = l.y & {ymask}u; let li = lid & {lmask}u;
    let row0 = g.y * BM; let col0 = g.x * 16u;
    {body}
    let col = col0 + tx;
{stores}}}
"#,
        at = 16 * rows,
        ymask = ty - 1,
        lmask = threads - 1,
    )
}

// y = r + (Σ partials in slice order + bias) for the partials of a split GEMM.
pub(super) fn reduce() -> String {
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> r: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let m = p.a.x; let n4 = p.a.y / 4u; let slices = p.a.z; let i = id.x;
    if (i >= m * n4) {{ return; }}
    var v = vec4<f32>(0.0);
    for (var s = 0u; s < slices; s++) {{ v = v + part[s * m * n4 + i]; }}
    y[i] = r[i] + (v + bias[i % n4]);
}}
"#
    )
}

// LayerNorm over one row per workgroup; `embed` repeats the same arithmetic.
pub(super) fn norm() -> String {
    let lanes = D / 4;
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: Params;
{TOTAL}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let i = g.x * {lanes}u + t;
    var v = vec4<f32>(0.0);
    if (t < {lanes}u) {{ v = a[i]; }}
    let mean = total((v.x + v.y) + (v.z + v.w), t) / {D}.0;
    var c = vec4<f32>(0.0);
    if (t < {lanes}u) {{ c = v - mean; }}
    let variance = total(dot(c, c), t) / {D}.0;
    let inv = 1.0 / sqrt(variance + bitcast<f32>(p.a.x));
    if (t < {lanes}u) {{ y[i] = c * inv * w[t] + bias[t]; }}
}}
"#
    )
}

// Word + type + position embeddings of each (token id, position) table pair, then the embedding
// LayerNorm, one row per workgroup in table order. The vocabulary is split into buffers of
// 65536 rows to fit storage binding limits; `pre` holds the type row, the norm weight and bias,
// then the position rows.
pub(super) fn embed() -> String {
    let lanes = D / 4;
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> v0: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> v1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v2: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v3: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pre: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tab: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> p: Params;
{TOTAL}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let base = p.a.x + g.x * 2u;
    let id = tab[base]; let pos = tab[base + 1u];
    let r = (id & 65535u) * {lanes}u + t;
    var v = vec4<f32>(0.0);
    if (t < {lanes}u) {{
        var w: vec4<f32>;
        switch (id >> 16u) {{
            case 0u: {{ w = v0[r]; }}
            case 1u: {{ w = v1[r]; }}
            case 2u: {{ w = v2[r]; }}
            default: {{ w = v3[r]; }}
        }}
        v = (w + pre[t]) + pre[{positions}u + pos * {lanes}u + t];
    }}
    let mean = total((v.x + v.y) + (v.z + v.w), t) / {D}.0;
    var c = vec4<f32>(0.0);
    if (t < {lanes}u) {{ c = v - mean; }}
    let variance = total(dot(c, c), t) / {D}.0;
    let inv = 1.0 / sqrt(variance + bitcast<f32>(p.a.y));
    if (t < {lanes}u) {{ y[g.x * {lanes}u + t] = c * inv * pre[{lanes}u + t] + pre[{bias}u + t]; }}
}}
"#,
        positions = 3 * lanes,
        bias = 2 * lanes,
    )
}

const LANES: [&str; 4] = ["x", "y", "z", "w"];
fn vector(n: usize) -> String {
    if n == 1 {
        "f32".into()
    } else {
        format!("vec{n}<f32>")
    }
}
fn splat(n: usize, s: &str) -> String {
    if n == 1 {
        s.into()
    } else {
        format!("{}({s})", vector(n))
    }
}
fn lane(n: usize, v: &str, j: usize) -> String {
    if n == 1 {
        v.into()
    } else {
        format!("{v}.{}", LANES[j])
    }
}

// Attention tiles: each row of the table is (first query, query count, first key, key count),
// so queries only see their own sequence. Registers are named scalars and vectors rather than
// dynamically indexed arrays, which WebKit would keep in memory.

// Row-per-thread attention for narrow heads: thread (query, split) keeps its query and output
// in registers and walks keys split, split + splits, … of each shared key tile with an online
// softmax; the partial results of a query are merged once, in split order, at the end.
pub(super) fn attention_rows(head: usize, queries: usize, splits: usize, keys: usize) -> String {
    let threads = queries * splits;
    let c4 = head / 4;
    let per = keys / splits;
    assert!(threads <= 256 && keys % splits == 0 && head % 4 == 0);
    let each = |n: usize, f: &dyn Fn(usize) -> String, sep: &str| {
        (0..n).map(f).collect::<Vec<_>>().join(sep)
    };
    let load_q = each(c4, &|c| format!("let q{c} = qb[row + {c}u] * scale;"), " ");
    let zero_o = each(c4, &|c| format!("var o{c} = vec4<f32>(0.0);"), " ");
    let scores = each(
        per,
        &|u| {
            let key = format!("split + {}u", u * splits);
            let dot = each(
                c4,
                &|c| format!("dot(q{c}, ks[({key}) * {c4}u + {c}u])"),
                " + ",
            );
            format!("var s{u} = {dot}; if (kt + {key} >= kn) {{ s{u} = -3.0e38; }}")
        },
        "\n        ",
    );
    let top = (1..per).fold("s0".to_string(), |a, u| format!("max({a}, s{u})"));
    let rescale = each(c4, &|c| format!("o{c} *= cc;"), " ");
    let accumulate = each(
        per,
        &|u| {
            let key = format!("split + {}u", u * splits);
            let fmas = each(
                c4,
                &|c| format!("o{c} = fma(vec4<f32>(e), vs[({key}) * {c4}u + {c}u], o{c});"),
                " ",
            );
            format!(
                "{{ var e = exp(s{u} - n); if (kt + {key} >= kn) {{ e = 0.0; }} l += e; {fmas} }}"
            )
        },
        "\n        ",
    );
    let finish = each(c4, &|c| format!("o{c} *= f;"), " ");
    let merge = each(
        c4,
        &|c| {
            format!(
                "os[{c}u * {queries}u + qi] = select(o{c}, os[{c}u * {queries}u + qi] + o{c}, s > 0u);"
            )
        },
        " ",
    );
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: Params;
var<workgroup> ks: array<vec4<f32>, {tile}>;
var<workgroup> vs: array<vec4<f32>, {tile}>;
var<workgroup> ml: array<vec2<f32>, {threads}>;
var<workgroup> os: array<vec4<f32>, {out}>;
@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let li = lid & {mask}u;
    let base = p.b.w + g.x * 4u;
    let first = t[base]; let qn = t[base + 1u]; let k0 = t[base + 2u]; let kn = t[base + 3u];
    let qi = li % {queries}u; let split = li / {queries}u;
    let scale = bitcast<f32>(p.b.z);
    let hc = g.y * {c4}u; let kv4 = p.a.z / 4u;
    // Spare threads of a short tile repeat the last query and store nothing.
    let row = min(first + qi, first + qn - 1u) * (p.a.x / 4u) + p.a.y / 4u + hc;
    {load_q}
    var m = -3.0e38; var l = 0.0;
    {zero_o}
    for (var kt = 0u; kt < kn; kt += {keys}u) {{
        workgroupBarrier();
        for (var i = li; i < {tile}u; i += {threads}u) {{
            let r = i / {c4}u; let c = i % {c4}u; var kk = vec4<f32>(0.0); var vv = kk;
            if (kt + r < kn) {{ let at = (k0 + kt + r) * kv4 + hc + c; kk = kv[at + p.a.w / 4u]; vv = kv[at + p.b.x / 4u]; }}
            ks[i] = kk; vs[i] = vv;
        }}
        workgroupBarrier();
        {scores}
        let n = max(m, {top});
        let cc = exp(m - n); l *= cc;
        {rescale}
        {accumulate}
        m = n;
    }}
    ml[li] = vec2<f32>(m, l);
    workgroupBarrier();
    var top = -3.0e38;
    for (var s = 0u; s < {splits}u; s++) {{ top = max(top, ml[s * {queries}u + qi].x); }}
    var total = 0.0;
    for (var s = 0u; s < {splits}u; s++) {{ let v = ml[s * {queries}u + qi]; total += v.y * exp(v.x - top); }}
    let f = exp(m - top) / total;
    {finish}
    for (var s = 0u; s < {splits}u; s++) {{
        if (split == s) {{ {merge} }}
        workgroupBarrier();
    }}
    for (var i = li; i < {out}u; i += {threads}u) {{
        let r = i / {c4}u; let c = i % {c4}u;
        if (r < qn) {{ y[(first + r) * (p.b.y / 4u) + hc + c] = os[c * {queries}u + r]; }}
    }}
}}
"#,
        tile = keys * c4,
        out = queries * c4,
        mask = threads - 1,
    )
}

// Tiled attention for wide heads: a thread scores a pair of queries against a slice of each
// key tile, then accumulates the pair over a slice of the head dimensions.
pub(super) fn attention_tiled(head: usize, queries: usize, keys: usize) -> String {
    let pairs = queries / 2;
    let groups = 128 / pairs;
    let kpt = keys / groups;
    let dpt = head / groups;
    let dw = dpt.min(4);
    let (dc, hdw) = (dpt / dw, head / dw);
    assert!(matches!(kpt, 1 | 2 | 4) && dpt >= 1 && pairs * groups == 128);
    let each = |n: usize, f: &dyn Fn(usize) -> String, sep: &str| {
        (0..n).map(f).collect::<Vec<_>>().join(sep)
    };
    let key = |j: usize| format!("kt + tg * {kpt}u + {j}u");
    let zero_s = each(kpt, &|j| format!("var s{j} = vec2<f32>(0.0);"), " ");
    let score = each(
        kpt,
        &|j| format!("s{j} = fma(qq, vec2<f32>({}), s{j});", lane(kpt, "kk", j)),
        " ",
    );
    let masked = each(
        kpt,
        &|j| format!("if ({} >= kn) {{ s{j} = vec2<f32>(-3.0e38); }}", key(j)),
        " ",
    );
    let top = each(kpt, &|j| format!("x = max(x, s{j});"), " ");
    let weights = each(
        kpt,
        &|j| {
            format!(
                "{{ var e = vec2<f32>(0.0); if ({k} < kn) {{ e = exp(s{j} - n); }} ps[(tg * {kpt}u + {j}u) * {pairs}u + qg] = e; sum += e; }}",
                k = key(j)
            )
        },
        "\n        ",
    );
    let load_k = each(
        kpt,
        &|j| {
            format!(
                "if (kt + kg * {kpt}u + {j}u < kn) {{ {} = kv[(k0 + kt + kg * {kpt}u + {j}u) * p.a.z + h * {head}u + d + p.a.w]; }}",
                lane(kpt, "kk", j)
            )
        },
        " ",
    );
    let load_v = if dw == 1 {
        "vv = kv[at];".to_string()
    } else {
        each(dw, &|j| format!("vv.{} = kv[at + {j}u];", LANES[j]), " ")
    };
    let mut zero_o = String::new();
    let mut rescale = String::new();
    let mut accumulate = String::new();
    let mut store = String::new();
    for c in 0..dc {
        accumulate += &format!("{{ let v = vs[key * {hdw}u + {}u + tg];", c * groups);
        for j in 0..dw {
            zero_o += &format!("var o{c}_{j} = vec2<f32>(0.0); ");
            rescale += &format!("o{c}_{j} *= cc; ");
            accumulate += &format!(
                " o{c}_{j} = fma(w, vec2<f32>({}), o{c}_{j});",
                lane(dw, "v", j)
            );
            for r in 0..2 {
                store += &format!(
                    "if (qg * 2u + {r}u < qn) {{ y[(first + qg * 2u + {r}u) * p.b.y + h * {head}u + ({}u + tg) * {dw}u + {j}u] = o{c}_{j}.{l} * inv.{l}; }}\n    ",
                    c * groups,
                    l = LANES[r]
                );
            }
        }
        accumulate += " }\n            ";
    }
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
// Query pairs dimension-major, keys dimension-major in groups of {kpt}, values key-major.
var<workgroup> qs: array<vec2<f32>, {qs}>;
var<workgroup> ks: array<{kv}, {ks}>;
var<workgroup> vs: array<{dv}, {vs}>;
var<workgroup> ps: array<vec2<f32>, {ps}>;
var<workgroup> rm: array<vec2<f32>, 128>;
var<workgroup> rl: array<vec2<f32>, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let li = lid & 127u;
    let base = p.b.w + g.x * 4u;
    let first = t[base]; let qn = t[base + 1u]; let k0 = t[base + 2u]; let kn = t[base + 3u];
    let h = g.y; let qg = li / {groups}u; let tg = li % {groups}u;
    let scale = bitcast<f32>(p.b.z);
    for (var i = li; i < {qs}u; i += 128u) {{
        let gq = i / {head}u; let d = i % {head}u; var v = vec2<f32>(0.0);
        if (gq * 2u < qn) {{ v.x = q[(first + gq * 2u) * p.a.x + p.a.y + h * {head}u + d]; }}
        if (gq * 2u + 1u < qn) {{ v.y = q[(first + gq * 2u + 1u) * p.a.x + p.a.y + h * {head}u + d]; }}
        qs[d * {pairs}u + gq] = v * scale;
    }}
    var m = vec2<f32>(-3.0e38); var l = vec2<f32>(0.0);
    {zero_o}
    for (var kt = 0u; kt < kn; kt += {keys}u) {{
        workgroupBarrier();
        for (var i = li; i < {kgroups}u; i += 128u) {{
            let kg = i / {head}u; let d = i % {head}u; var kk = {kzero};
            {load_k}
            ks[d * {groups}u + kg] = kk;
        }}
        for (var i = li; i < {vs}u; i += 128u) {{
            let r = i / {hdw}u; let c = i % {hdw}u; var vv = {vzero};
            if (kt + r < kn) {{
                let at = (k0 + kt + r) * p.a.z + h * {head}u + c * {dw}u + p.b.x;
                {load_v}
            }}
            vs[i] = vv;
        }}
        workgroupBarrier();
        {zero_s}
        for (var d = 0u; d < {head}u; d++) {{
            let qq = qs[d * {pairs}u + qg]; let kk = ks[d * {groups}u + tg];
            {score}
        }}
        {masked}
        var x = vec2<f32>(-3.0e38);
        {top}
        rm[tg * {pairs}u + qg] = x;
        workgroupBarrier();
        var n = m;
        for (var i = 0u; i < {groups}u; i++) {{ n = max(n, rm[i * {pairs}u + qg]); }}
        let cc = exp(m - n);
        var sum = vec2<f32>(0.0);
        {weights}
        rl[tg * {pairs}u + qg] = sum;
        workgroupBarrier();
        var u = vec2<f32>(0.0);
        for (var i = 0u; i < {groups}u; i++) {{ u += rl[i * {pairs}u + qg]; }}
        l = l * cc + u;
        {rescale}
        let limit = min({keys}u, kn - kt);
        for (var key = 0u; key < limit; key++) {{
            let w = ps[key * {pairs}u + qg];
            {accumulate}
        }}
        m = n;
    }}
    let inv = 1.0 / l;
    {store}
}}
"#,
        qs = head * pairs,
        ks = head * groups,
        kgroups = head * groups,
        vs = keys * hdw,
        ps = keys * pairs,
        kv = vector(kpt),
        dv = vector(dw),
        kzero = splat(kpt, "0.0"),
        vzero = splat(dw, "0.0"),
    )
}

// Workgroup memory the tiled attention kernel needs, in bytes.
pub(super) fn attention_tiled_storage(head: usize, queries: usize, keys: usize) -> usize {
    4 * (head * queries + 2 * head * keys + keys * queries + 512)
}

// Mean of each (first row, row count, output row) segment.
pub(super) fn mean() -> String {
    let lanes = D / 4;
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size({lanes})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let base = p.a.x + g.x * 3u;
    let first = t[base]; let count = t[base + 1u]; let out = t[base + 2u];
    var sum = vec4<f32>(0.0);
    for (var r = 0u; r < count; r++) {{ sum += x[(first + r) * {lanes}u + li]; }}
    y[out * {lanes}u + li] = sum / f32(count);
}}
"#
    )
}

// Copies rows by (source row, destination row) pairs.
pub(super) fn copy() -> String {
    let lanes = D / 4;
    format!(
        r#"{PARAMS}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size({lanes})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let base = p.a.x + g.x * 2u;
    y[t[base + 1u] * {lanes}u + li] = x[t[base] * {lanes}u + li];
}}
"#
    )
}
