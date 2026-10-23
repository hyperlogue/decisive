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
export const categories = [
  {id:'name', icon:'person', severity:'mild', question:'name'},
  {id:'birthday', icon:'calendar', severity:'mild', question:'personal'},
  {id:'contact', icon:'mail', severity:'mild', question:'personal'},
  {id:'precise', icon:'home', severity:'strong', question:'personal'},
  {id:'financial', icon:'credit', severity:'strong', question:'financial'},
  {id:'identity', icon:'id', severity:'strong', question:'personal'},
  {id:'password', icon:'lock', severity:'strong', question:'secret'},
  {id:'token', icon:'key', severity:'strong', question:'secret'},
  {id:'otp', icon:'otp', severity:'strong', question:'secret'},
  {id:'privatekey', icon:'keyfile', severity:'strong', question:'secret'},
];
export const threshold = 0.5;
export const welcome = 'Welcome to the community. Tell us about yourself.';

export function buildRequest(draft, engine = 'decima') {
  return {
    state: {
      context: [{role:'assistant', content:welcome}, {role:'user', content:draft}],
    },
    questions,
    ...(engine==='decima'?{long:true}:{}),
  };
}

export function readScores(response) {
  return categories.map(c => {
    const answer = response?.answers?.[c.question];
    const score = answer?.type !== questions[c.question].type ? undefined
      : answer.type === 'noul' ? answer.noul : (answer.max_probabilities ?? answer.probabilities)?.[c.id];
    if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('invalidOutput');
    return {...c, score, detected:score >= threshold};
  });
}
