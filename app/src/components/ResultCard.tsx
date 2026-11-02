import { useEffect } from 'react';
import { CONDITION_INDEX, CROP_INDEX, OTHER_ID, type Lang } from '../data/crops';
import { UI } from '../data/ui';
import { addToDiary } from '../game/sim';
import type { Diagnosis } from '../lib/classifier';
import { sfx } from '../lib/sound';
import { speak, ttsSupported } from '../lib/tts';
import { Farmer } from './art/Farmer';

const MIN_CONFIDENCE = 0.45;

interface Props {
  lang: Lang;
  image: string;
  diagnosis: Diagnosis;
  demo: boolean;
  onReset(): void;
}

export function ResultCard({ lang, image, diagnosis, demo, onReset }: Props) {
  const { predictions, crops, matches } = diagnosis;
  // Always name the closest crop condition; low confidence or "other" only adds a caution.
  const top = predictions.find((p) => CONDITION_INDEX.has(p.id)) ?? predictions[0];
  const hit = CONDITION_INDEX.get(top.id);
  const unsupported = crops[0].id === OTHER_ID && crops[0].score >= 0.5;
  const sure = !!hit && !unsupported && top.score >= MIN_CONFIDENCE;
  const topCrop = crops.find((c) => c.id === hit?.crop.id) ?? crops[0];
  const cropInfo = CROP_INDEX.get(topCrop.id);
  const caution = unsupported ? UI.notSupported[lang] : sure ? null : UI.lowConfidence[lang];

  useEffect(() => {
    if (!sure) return;
    if (hit.condition.healthy) sfx.levelUp();
    else {
      sfx.sick();
      if (!demo) addToDiary(hit.condition.id);
    }
  }, [sure, hit, demo]);

  function message() {
    if (!hit) return UI.notSupported[lang];
    const { crop, condition } = hit;
    return [
      ...(caution ? [caution] : []),
      `${crop.name[lang]}: ${condition.name[lang]}.`,
      `${UI.symptoms[lang]}: ${condition.symptoms[lang]}`,
      `${UI.treatment[lang]}: ${condition.treatment[lang]}`,
    ].join(' ');
  }

  return (
    <div className="result">
      {demo && <p className="demo-banner">{UI.demo[lang]}</p>}
      <div className="shot-wrap">
        <img className="shot" src={image} alt="" />
        {cropInfo && (
          <span className="crop-tag">
            {UI.cropFound[lang]}: {cropInfo.emoji} {cropInfo.name[lang]} · {Math.round(topCrop.score * 100)}%
          </span>
        )}
      </div>

      {hit ? (
        <div className={`verdict ${hit.condition.healthy ? 'ok' : 'bad'}`}>
          {caution && <p className="caution">⚠️ {caution}</p>}
          <div className="verdict-head">
            <Farmer size={48} mood={hit.condition.healthy ? 'cheer' : 'worried'} />
            <div>
              <h2>{hit.condition.healthy ? '✅' : '⚠️'} {hit.condition.name[lang]}</h2>
              <div className="meter leaf" aria-label={UI.confidence[lang]}>
                <div style={{ width: `${Math.round(top.score * 100)}%` }} />
              </div>
              <small>{UI.confidence[lang]}: {Math.round(top.score * 100)}%</small>
            </div>
          </div>
          <h3>{UI.symptoms[lang]}</h3>
          <p>{hit.condition.symptoms[lang]}</p>
          <h3>{UI.treatment[lang]}</h3>
          <p>{hit.condition.treatment[lang]}</p>
        </div>
      ) : (
        <div className="verdict unsure">
          <div className="verdict-head">
            <Farmer size={48} mood="worried" />
            <p>{UI.notSupported[lang]}</p>
          </div>
        </div>
      )}

      {matches.length > 0 && (
        <section className="matches">
          <h3>{UI.similar[lang]}</h3>
          <div className="match-grid">
            {matches.map((m) => {
              const info = CONDITION_INDEX.get(m.id);
              return (
                <figure key={m.img}>
                  <img src={m.img} alt="" loading="lazy" />
                  <figcaption>
                    {info ? `${info.crop.emoji} ${info.condition.name[lang]}` : `🌿 ${UI.otherPlant[lang]}`}
                    <small>{Math.round(Math.max(0, m.similarity) * 100)}%</small>
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </section>
      )}

      {predictions.length > 1 && (
        <details>
          <summary>{UI.otherMatches[lang]}</summary>
          <ul>
            {predictions.filter((p) => p !== top).map((p) => {
              const h = CONDITION_INDEX.get(p.id);
              return (
                <li key={p.id}>
                  {h ? `${h.crop.name[lang]}: ${h.condition.name[lang]}` : `🌿 ${UI.otherPlant[lang]}`} ({Math.round(p.score * 100)}%)
                </li>
              );
            })}
          </ul>
        </details>
      )}

      <div className="actions">
        {ttsSupported && <button className="primary" onClick={() => speak(message(), lang)}>🔊 {UI.listen[lang]}</button>}
        <button className="secondary" onClick={onReset}>🔄 {UI.retake[lang]}</button>
      </div>
      <p className="hint small">{UI.disclaimer[lang]}</p>
    </div>
  );
}
