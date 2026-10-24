export const questions = {
  name:{type:'noul', instructions:'Does the user’s draft disclose a real personal name?'},
  personal:{type:'choice', instructions:'Which personal detail does the user’s draft give?', criteria:{
    birthday:'a date of birth', contact:'a personal email address or phone number',
    precise:'a home address or exact current location', identity:'a government ID number',
    fake:'only a public, business, or example detail', none:'no personal details'}},
  financial:{type:'noul', instructions:'Does the user’s draft reveal a person’s bank, card, income, or debt details?'},
  secret:{type:'choice', instructions:'Which secret value does the user’s draft contain?', criteria:{
    password:'password or PIN', token:'API key, access token, or session cookie', otp:'one-time verification code',
    privatekey:'private key or wallet recovery phrase', fake:'only a placeholder, example, or public value', none:'no secret value'}},
};
const pplxQuestions = {
  ...questions,
  secret:{...questions.secret, criteria:{
    credential:'password, PIN, one-time verification code, API key, access token, or session cookie',
    privatekey:questions.secret.criteria.privatekey, fake:questions.secret.criteria.fake, none:questions.secret.criteria.none}},
};
const isPplx = engine => engine?.startsWith('pplx-');
const questionsFor = engine => isPplx(engine) ? pplxQuestions : questions;
export const categories = [
  {id:'name', icon:'person', severity:'mild', question:'name'},
  {id:'birthday', icon:'calendar', severity:'mild', question:'personal'},
  {id:'contact', icon:'mail', severity:'mild', question:'personal'},
  {id:'precise', icon:'home', severity:'strong', question:'personal'},
  {id:'financial', icon:'credit', severity:'strong', question:'financial'},
  {id:'identity', icon:'id', severity:'strong', question:'personal'},
  {id:'credential', icon:'lock', severity:'strong', question:'secret'},
  {id:'privatekey', icon:'keyfile', severity:'strong', question:'secret'},
];
export const threshold = 0.5;
export const welcome = 'Welcome to the community. Tell us about yourself.';

export function buildRequest(draft, engine = 'decima') {
  return {
    state: {
      context: [{role:'assistant', content:welcome}, {role:'user', content:draft}],
    },
    questions:questionsFor(engine),
  };
}

export function readScores(response, engine = 'decima') {
  return categories.map(c => {
    const answer = response?.answers?.[c.question];
    if (answer?.type !== questionsFor(engine)[c.question].type) throw new Error('invalidOutput');
    const options = c.id === 'credential' && !isPplx(engine) ? ['password','token','otp'] : [c.id];
    const probabilities = answer.max_probabilities ?? answer.probabilities;
    const values = answer.type === 'noul' ? [answer.noul] : options.map(id=>probabilities?.[id]);
    if (values.some(value=>!Number.isFinite(value) || value < 0 || value > 1)) throw new Error('invalidOutput');
    const score = Math.max(...values);
    return {...c, score, detected:score >= threshold};
  });
}
