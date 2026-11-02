import type { Text } from './crops';

const t = (en: string, mr: string): Text => ({ en, mr });

export const UI = {
  title: t('Plant Doctor', 'पीक डॉक्टर'),
  subtitle: t('Your friendly farm helper', 'तुमचा शेतीतील मित्र'),
  greeting: t(
    'Namaskar! I am Shetkari Dada. Show me a sick leaf, or come grow a crop with me!',
    'नमस्कार! मी शेतकरी दादा. आजारी पान दाखवा, किंवा माझ्यासोबत पीक वाढवा!',
  ),
  scanTitle: t('Check a leaf', 'पान तपासा'),
  scanDesc: t('Take a photo, find the disease', 'फोटो काढा, रोग ओळखा'),
  growTitle: t('Grow & learn', 'वाढवा आणि शिका'),
  growDesc: t('A farming game about weather and disease', 'हवामान आणि रोगांचा शेती खेळ'),
  diaryTitle: t('Disease diary', 'रोग डायरी'),
  diaryDesc: t('Diseases you have discovered', 'तुम्ही शोधलेले रोग'),
  back: t('Back', 'मागे'),
  home: t('Home', 'मुख्यपृष्ठ'),

  // scanner
  openCamera: t('Open camera', 'कॅमेरा उघडा'),
  capture: t('Take photo', 'फोटो घ्या'),
  upload: t('Choose photo', 'फोटो निवडा'),
  retake: t('Check another leaf', 'दुसरे पान तपासा'),
  analysing: t('Looking closely at the leaf…', 'पान नीट पाहत आहे…'),
  listen: t('Listen', 'ऐका'),
  symptoms: t('Symptoms', 'लक्षणे'),
  treatment: t('What to do', 'उपाय'),
  confidence: t('Confidence', 'खात्री'),
  cropFound: t('Leaf type', 'पानाचा प्रकार'),
  otherMatches: t('Other possibilities', 'इतर शक्यता'),
  similar: t('Closest matches in our database', 'आमच्या संग्रहातील सर्वात जुळणारे फोटो'),
  otherPlant: t('Other plant', 'इतर वनस्पती'),
  supported: t('Crops I know', 'मला माहित असलेली पिके'),
  scanTip: t('Tip: one leaf, filling the frame, in daylight.', 'सूचना: एकच पान, फ्रेम भरून, दिवसाच्या प्रकाशात.'),
  demo: t(
    'DEMO MODE: no trained model installed, results are random.',
    'डेमो मोड: प्रशिक्षित मॉडेल नाही, निकाल यादृच्छिक आहेत.',
  ),
  lowConfidence: t(
    'This is my best guess, but I am not fully sure. A photo of one leaf filling the frame, in daylight, will help.',
    'हा माझा सर्वात जवळचा अंदाज आहे, पण पूर्ण खात्री नाही. दिवसाच्या प्रकाशात एकच पान फ्रेम भरून फोटो घेतल्यास मदत होईल.',
  ),
  notSupported: t(
    'This may not be cotton, soybean, potato, rice or tomato, the five crops I know. Here is the closest match.',
    'हे कापूस, सोयाबीन, बटाटा, भात किंवा टोमॅटो (मला माहित असलेली पाच पिके) नसावे. हा सर्वात जवळचा अंदाज आहे.',
  ),
  noCamera: t('Camera not available. Choose a photo instead.', 'कॅमेरा उपलब्ध नाही. त्याऐवजी फोटो निवडा.'),
  disclaimer: t(
    'This is a helper, not a lab test. Confirm with your Krishi Vigyan Kendra before spraying.',
    'हे मदतीसाठी आहे, प्रयोगशाळा चाचणी नाही. फवारणीपूर्वी कृषी विज्ञान केंद्राचा सल्ला घ्या.',
  ),

  // game
  gameIntro: t(
    'Pick a crop and a season. Change the weather and see which diseases come. Keep the plant healthy till harvest!',
    'पीक आणि हंगाम निवडा. हवामान बदला आणि कोणते रोग येतात ते पाहा. काढणीपर्यंत झाड निरोगी ठेवा!',
  ),
  pickCrop: t('Choose your crop', 'पीक निवडा'),
  pickSeason: t('Choose the season', 'हंगाम निवडा'),
  seasonHint: t('You can change the weather any time while playing.', 'खेळताना तुम्ही कधीही हवामान बदलू शकता.'),
  plantSeed: t('Plant the seed', 'बी पेरा'),
  days: t('days', 'दिवस'),
  day: t('Day', 'दिवस'),
  growth: t('Growth', 'वाढ'),
  health: t('Health', 'आरोग्य'),
  soil: t('Soil water', 'मातीतील ओलावा'),
  actions: t('Farm work', 'शेतीची कामे'),
  free: t('free', 'मोफत'),
  weather: t('Weather', 'हवामान'),
  temperature: t('Temperature', 'तापमान'),
  humidity: t('Humidity', 'आर्द्रता'),
  rain: t('Rain', 'पाऊस'),
  rainLevels: t('None|Drizzle|Rain|Heavy', 'नाही|रिमझिम|पाऊस|मुसळधार'),
  fertiliser: t('Nitrogen fertiliser', 'नत्र खत'),
  nLevels: t('Low|Normal|High', 'कमी|योग्य|जास्त'),
  diseaseRisk: t('Disease watch', 'रोग निरीक्षण'),
  riskHint: t(
    'The bar fills up when the weather suits a disease. Tap a disease to learn why.',
    'हवामान रोगाला पोषक असेल तर पट्टी भरते. कारण जाणून घेण्यासाठी रोगावर टॅप करा.',
  ),
  riskLevels: t('low|medium|high', 'कमी|मध्यम|जास्त'),
  harvestTitle: t('Harvest time! 🎉', 'काढणीची वेळ! 🎉'),
  lostTitle: t('The crop was lost 😢', 'पीक वाया गेले 😢'),
  yieldLabel: t('Yield', 'उत्पादन'),
  metDiseases: t('Diseases this season', 'या हंगामातील रोग'),
  none: t('none, well done!', 'एकही नाही, शाब्बास!'),
  lessonTip: t(
    'Try the same crop in another season and watch how the diseases change.',
    'हेच पीक दुसऱ्या हंगामात लावून पाहा, रोग कसे बदलतात ते बघा.',
  ),
  playAgain: t('Play again', 'पुन्हा खेळा'),
  otherCrop: t('Another crop', 'दुसरे पीक'),

  // diary
  diaryIntro: t(
    'Every disease you meet in the game or find with the camera is saved here.',
    'खेळात किंवा कॅमेऱ्याने सापडलेला प्रत्येक रोग इथे जतन होतो.',
  ),
  locked: t('Not discovered yet', 'अजून शोधलेला नाही'),
  found: t('found', 'सापडले'),
  sound: t('Sound', 'आवाज'),
};
