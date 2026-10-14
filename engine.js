/* MailGuard AI inference engine.
   JS port of src/train_model.py (clean_text) + sklearn TfidfVectorizer(1,2) + MultinomialNB.
   All numbers come from docs/model.js, which is exported by the trained Python model. */
(function (root) {
  function create(M) {
    const stop = new Set(M.stop);
    const idx = new Map(M.vocab.map((w, i) => [w, i]));
    const spamI = M.classes.indexOf("spam"), hamI = M.classes.indexOf("ham");
    const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;

    function clean(text) {
      return text.toLowerCase()
        .replace(/http\S+|www\.\S+/g, " urltoken ")
        .replace(/\S+@\S+/g, " emailtoken ")
        .replace(/\b\d{5,}\b/g, " longnumtoken ")
        .replace(/\d+/g, " numtoken ")
        .replace(PUNCT, "")
        .split(/\s+/).filter(t => t.length > 1 && !stop.has(t)).join(" ");
    }

    function vectorize(cleaned) {
      const toks = cleaned.match(/[\p{L}\p{N}_]{2,}/gu) || [];
      const counts = new Map();
      const add = t => { const i = idx.get(t); if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1); };
      toks.forEach((t, k) => { add(t); if (k) add(toks[k - 1] + " " + t); });
      let norm = 0;
      const x = new Map();
      counts.forEach((c, i) => { const v = c * M.idf[i]; x.set(i, v); norm += v * v; });
      norm = Math.sqrt(norm) || 1;
      x.forEach((v, i) => x.set(i, v / norm));
      return x;
    }

    function classify(subject, body) {
      const raw = ((subject || "") + " " + (body || "")).trim();
      if (!raw) throw new Error("EMPTY");
      const cleaned = clean(raw);
      const x = vectorize(cleaned);
      if (!cleaned || !x.size) throw new Error("NO_FEATURES");
      const jll = M.classes.map((_, c) => {
        let s = M.prior[c];
        x.forEach((v, i) => { s += v * M.lp[c][i]; });
        return s;
      });
      const mx = Math.max(...jll), e = jll.map(v => Math.exp(v - mx)), z = e.reduce((a, b) => a + b, 0);
      const spam = (e[spamI] / z) * 100, ham = (e[hamI] / z) * 100;
      // Per-term evidence: tf-idf weight x (log P(term|spam) - log P(term|ham))
      const terms = [];
      x.forEach((v, i) => terms.push({ term: M.vocab[i], score: v * (M.lp[spamI][i] - M.lp[hamI][i]) }));
      terms.sort((a, b) => b.score - a.score);
      return {
        isSpam: spam >= ham, spam, ham, confidence: Math.max(spam, ham),
        spamTerms: terms.filter(t => t.score > 0).slice(0, 6),
        hamTerms: terms.filter(t => t.score < 0).reverse().slice(0, 6),
      };
    }
    return { clean, classify };
  }
  if (typeof module !== "undefined") module.exports = { create }; else root.MailGuardEngine = { create };
})(this);
