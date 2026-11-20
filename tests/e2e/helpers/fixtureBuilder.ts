import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import type { FixtureEntry, Manifest } from './types';

const REAL_GROCERY_SRC =
  'C:/Users/paste/.gemini/antigravity/brain/ac206c0e-4b70-4ead-8c0c-e5de2cc382b1/.user_uploaded/media_1795025438054_0b38e6ea.png';

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

async function svgToPng(svg: string, outPath: string): Promise<void> {
  await sharp(Buffer.from(svg, 'utf8')).png().toFile(outPath);
}

function makeTextSvg(opts: {
  width: number;
  height: number;
  bg?: string;
  fg?: string;
  fontFamily: string;
  fontSize: number;
  fontWeight?: string;
  lines: string[];
  startX?: number;
  startY?: number;
  lineHeight?: number;
  skewXDeg?: number;
  translateX?: number;
  ruledLines?: boolean;
  spiralLeft?: boolean;
  topRightSmudge?: boolean;
  extraSvg?: string;
}): string {
  const bg = opts.bg ?? '#ffffff';
  const fg = opts.fg ?? '#18181b';
  const startX = opts.startX ?? 36;
  const startY = opts.startY ?? Math.round(opts.fontSize * 1.45);
  const lh = opts.lineHeight ?? Math.round(opts.fontSize * 1.45);
  const weight = opts.fontWeight ?? 'normal';

  const ruledSvg = opts.ruledLines
    ? opts.lines
        .map((_, idx) => {
          const ly = startY + idx * lh + Math.round(opts.fontSize * 0.22);
          return `<line x1="12" y1="${ly}" x2="${opts.width - 12}" y2="${ly}" stroke="#93c5fd" stroke-width="1.5" />`;
        })
        .join('\n')
    : '';

  const spiralSvg = opts.spiralLeft
    ? Array.from({ length: 7 }, (_, i) => {
        const cy = 28 + i * Math.floor((opts.height - 56) / 6);
        return `<circle cx="14" cy="${cy}" r="7" fill="#111111" /><rect x="0" y="${cy - 2}" width="14" height="4" fill="#111111" />`;
      }).join('\n')
    : '';

  const smudgeSvg = opts.topRightSmudge
    ? `<path d="M ${opts.width - 28} 0 L ${opts.width} 0 L ${opts.width} 30 Q ${opts.width - 18} 20 ${opts.width - 28} 0 Z" fill="#18181b" />`
    : '';

  const textEls = opts.lines
    .map((line, idx) => {
      const y = startY + idx * lh;
      return `<text x="${startX}" y="${y}" font-family="${escapeXml(opts.fontFamily)}" font-size="${opts.fontSize}" font-weight="${weight}" fill="${fg}">${escapeXml(line)}</text>`;
    })
    .join('\n');

  const transformAttr =
    opts.skewXDeg || opts.translateX
      ? ` transform="translate(${opts.translateX ?? 0},0) skewX(${opts.skewXDeg ?? 0})"`
      : '';

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${opts.width}" height="${opts.height}">
  <rect width="100%" height="100%" fill="${bg}" />
  ${ruledSvg}
  ${spiralSvg}
  ${smudgeSvg}
  ${opts.extraSvg ?? ''}
  <g${transformAttr}>
    ${textEls}
  </g>
</svg>`;
}

export async function generateAllFixtures(fixturesDir: string): Promise<Manifest> {
  fs.mkdirSync(fixturesDir, { recursive: true });

  const entries: Record<string, FixtureEntry> = {};

  // 1. Copy real grocery note verbatim
  const realDest = path.join(fixturesDir, 'real-grocery-note.png');
  if (fs.existsSync(REAL_GROCERY_SRC)) {
    fs.copyFileSync(REAL_GROCERY_SRC, realDest);
  } else if (!fs.existsSync(realDest)) {
    throw new Error(`Real grocery note source missing: ${REAL_GROCERY_SRC}`);
  }
  entries['real-grocery-note.png'] = {
    filename: 'real-grocery-note.png',
    groundTruth:
      'bread\nmilk\ncrisps\neggs\npan au chocolat\nlunch for tomorrow\nsalmon\nasparagus\nbacon\nturkey\nspinach\nturkey mince\nsundried tomatoes\nbrowlli',
    expectedStyle: 'handwriting',
    expectedLineCount: 14,
    tags: ['real-world', 'cursive', 'spiral', 'grocery'],
    description: 'User-uploaded 14-line handwritten grocery note with left spiral holes and top-right edge bleed',
  };

  // 2. Clean 2-line cursive
  await svgToPng(
    makeTextSvg({
      width: 540,
      height: 180,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['buy milk', 'call dentist'],
      startX: 40,
      startY: 64,
      lineHeight: 68,
    }),
    path.join(fixturesDir, 'cursive-clean-2lines.png')
  );
  entries['cursive-clean-2lines.png'] = {
    filename: 'cursive-clean-2lines.png',
    groundTruth: 'buy milk\ncall dentist',
    expectedStyle: 'handwriting',
    expectedLineCount: 2,
    tags: ['cursive', 'clean'],
    description: 'Clean 2-line cursive note',
  };

  // 3. Long single-line cursive (>48 tokens / 10+ words)
  await svgToPng(
    makeTextSvg({
      width: 1480,
      height: 120,
      fontFamily: 'Segoe Print',
      fontSize: 32,
      lines: ['Remember to bring passports tickets charger and laptop before nine'],
      startX: 30,
      startY: 72,
    }),
    path.join(fixturesDir, 'cursive-long-line.png')
  );
  entries['cursive-long-line.png'] = {
    filename: 'cursive-long-line.png',
    groundTruth: 'Remember to bring passports tickets charger and laptop before nine',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', 'long-line'],
    description: 'Long 10-word cursive line testing max_new_tokens 64 and splitWideLineBox',
  };

  // 4. Slanted +22 deg
  await svgToPng(
    makeTextSvg({
      width: 620,
      height: 130,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['meeting tomorrow'],
      startX: 60,
      startY: 80,
      skewXDeg: -22,
      translateX: 45,
    }),
    path.join(fixturesDir, 'cursive-slanted-pos22.png')
  );
  entries['cursive-slanted-pos22.png'] = {
    filename: 'cursive-slanted-pos22.png',
    groundTruth: 'meeting tomorrow',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', 'slanted', 'pos22'],
    description: 'Forward +22 deg slanted cursive text',
  };

  // 5. Slanted -18 deg
  await svgToPng(
    makeTextSvg({
      width: 580,
      height: 130,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['finish report'],
      startX: 60,
      startY: 80,
      skewXDeg: 18,
      translateX: 10,
    }),
    path.join(fixturesDir, 'cursive-slanted-neg18.png')
  );
  entries['cursive-slanted-neg18.png'] = {
    filename: 'cursive-slanted-neg18.png',
    groundTruth: 'finish report',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', 'slanted', 'neg18'],
    description: 'Backward -18 deg slanted cursive text',
  };

  // 6. Extreme +35 deg
  await svgToPng(
    makeTextSvg({
      width: 660,
      height: 150,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['morning coffee'],
      startX: 40,
      startY: 90,
      skewXDeg: -35,
      translateX: 85,
    }),
    path.join(fixturesDir, 'cursive-slanted-pos35.png')
  );
  entries['cursive-slanted-pos35.png'] = {
    filename: 'cursive-slanted-pos35.png',
    groundTruth: 'morning coffee',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', 'slanted', 'pos35', 'boundary'],
    description: 'Extreme +35 deg slanted cursive text',
  };

  // 7. Extreme -35 deg
  await svgToPng(
    makeTextSvg({
      width: 660,
      height: 150,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['evening walk'],
      startX: 90,
      startY: 90,
      skewXDeg: 35,
      translateX: -10,
    }),
    path.join(fixturesDir, 'cursive-slanted-neg35.png')
  );
  entries['cursive-slanted-neg35.png'] = {
    filename: 'cursive-slanted-neg35.png',
    groundTruth: 'evening walk',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', 'slanted', 'neg35', 'boundary'],
    description: 'Extreme -35 deg slanted cursive text',
  };

  // 8. Tight descenders/ascenders (f,g,j,l,p,q,y)
  await svgToPng(
    makeTextSvg({
      width: 560,
      height: 210,
      fontFamily: 'Segoe Print',
      fontSize: 38,
      lines: ['juggling playful', 'flying gray quails', 'jumping quietly'],
      startX: 36,
      startY: 54,
      lineHeight: 50,
    }),
    path.join(fixturesDir, 'cursive-tight-descenders.png')
  );
  entries['cursive-tight-descenders.png'] = {
    filename: 'cursive-tight-descenders.png',
    groundTruth: 'juggling playful\nflying gray quails\njumping quietly',
    expectedStyle: 'handwriting',
    expectedLineCount: 3,
    tags: ['cursive', 'ascenders', 'descenders'],
    description: '3 tightly spaced cursive lines rich in f,g,j,l,p,q,y ascenders/descenders',
  };

  // 9. Spiral holes + top-right smudge + 3 valid lines
  await svgToPng(
    makeTextSvg({
      width: 540,
      height: 280,
      fontFamily: 'Segoe Print',
      fontSize: 38,
      lines: ['buy apples', 'call mom', 'pay rent'],
      startX: 72,
      startY: 72,
      lineHeight: 74,
      spiralLeft: true,
      topRightSmudge: true,
    }),
    path.join(fixturesDir, 'cursive-spiral-bleed.png')
  );
  entries['cursive-spiral-bleed.png'] = {
    filename: 'cursive-spiral-bleed.png',
    groundTruth: 'buy apples\ncall mom\npay rent',
    expectedStyle: 'handwriting',
    expectedLineCount: 3,
    tags: ['cursive', 'spiral', 'bleed'],
    description: 'Notebook page with left spiral holes and top-right corner bleed plus 3 valid lines',
  };

  // 10. Wide empty margin + small text (triggers dpi-upscale + auto-bbox-shrink)
  await svgToPng(
    makeTextSvg({
      width: 820,
      height: 680,
      fontFamily: 'Segoe Print',
      fontSize: 15,
      lines: ['buy fresh bread', 'call dentist'],
      startX: 48,
      startY: 42,
      lineHeight: 26,
    }),
    path.join(fixturesDir, 'cursive-wide-margin.png')
  );
  entries['cursive-wide-margin.png'] = {
    filename: 'cursive-wide-margin.png',
    groundTruth: 'buy fresh bread\ncall dentist',
    expectedStyle: 'handwriting',
    expectedLineCount: 2,
    tags: ['cursive', 'dpi-upscale', 'auto-bbox-shrink'],
    description: 'Small-font note with large empty margin to trigger dpi-upscale and auto-bbox-shrink',
  };

  // 11. Cursive confusions + proper nouns + currency
  await svgToPng(
    makeTextSvg({
      width: 600,
      height: 240,
      fontFamily: 'Segoe Print',
      fontSize: 36,
      lines: ['morning milk', 'work date', 'Dr. Alice $45.50'],
      startX: 36,
      startY: 60,
      lineHeight: 68,
    }),
    path.join(fixturesDir, 'cursive-confusions.png')
  );
  entries['cursive-confusions.png'] = {
    filename: 'cursive-confusions.png',
    groundTruth: 'morning milk\nwork date\nDr. Alice $45.50',
    expectedStyle: 'handwriting',
    expectedLineCount: 3,
    tags: ['cursive', 'confusions', 'proper-nouns', 'currency'],
    description: 'Cursive confusion pairs with proper nouns and currency',
  };

  // 12. Grocery anchor preservation fixture
  await svgToPng(
    makeTextSvg({
      width: 520,
      height: 400,
      fontFamily: 'Segoe Print',
      fontSize: 36,
      lines: ['tomorrow', 'tomatoes', 'turkey', 'broccoli', 'salmon', 'pan au chocolat'],
      startX: 40,
      startY: 56,
      lineHeight: 60,
    }),
    path.join(fixturesDir, 'cursive-anchors-grocery.png')
  );
  entries['cursive-anchors-grocery.png'] = {
    filename: 'cursive-anchors-grocery.png',
    groundTruth: 'tomorrow\ntomatoes\nturkey\nbroccoli\nsalmon\npan au chocolat',
    expectedStyle: 'handwriting',
    expectedLineCount: 6,
    tags: ['cursive', 'anchors', 'grocery'],
    description: 'Grocery anchor words that must not be over-corrected',
  };

  // 13. Slanted handwritten meeting note on ruled paper (Tier 4 S2)
  await svgToPng(
    makeTextSvg({
      width: 820,
      height: 280,
      fontFamily: 'Segoe Print',
      fontSize: 32,
      lines: [
        'Meet Dr. Alice Bramwell on 5/12',
        'Review Q3 budget $450.00 at 3:30pm',
        'Bring Sarah Patel notebook',
      ],
      startX: 48,
      startY: 72,
      lineHeight: 74,
      skewXDeg: -20,
      translateX: 44,
      ruledLines: true,
    }),
    path.join(fixturesDir, 'meeting-note-slanted.png')
  );
  entries['meeting-note-slanted.png'] = {
    filename: 'meeting-note-slanted.png',
    groundTruth:
      'Meet Dr. Alice Bramwell on 5/12\nReview Q3 budget $450.00 at 3:30pm\nBring Sarah Patel notebook',
    expectedStyle: 'handwriting',
    expectedLineCount: 3,
    tags: ['scenario', 'slanted', 'ruled', 'proper-nouns'],
    description: 'Slanted handwritten meeting note on ruled paper',
  };

  // 14. Printed schedule
  await svgToPng(
    makeTextSvg({
      width: 600,
      height: 260,
      fontFamily: 'Arial',
      fontSize: 34,
      fontWeight: '600',
      lines: [
        'Dentist Appointment',
        'Tuesday 10:30am',
        'Bring Insurance Card',
        'Copay $35.00',
      ],
      startX: 36,
      startY: 56,
      lineHeight: 54,
    }),
    path.join(fixturesDir, 'printed-schedule.png')
  );
  entries['printed-schedule.png'] = {
    filename: 'printed-schedule.png',
    groundTruth: 'Dentist Appointment\nTuesday 10:30am\nBring Insurance Card\nCopay $35.00',
    expectedStyle: 'printed',
    expectedLineCount: 4,
    tags: ['printed', 'schedule'],
    description: 'Clean printed schedule note',
  };

  // 15. Dark-mode printed (>5% bright ratio)
  await svgToPng(
    makeTextSvg({
      width: 540,
      height: 210,
      bg: '#111318',
      fg: '#f4f4f5',
      fontFamily: 'Arial',
      fontSize: 38,
      fontWeight: 'bold',
      lines: ['Team Standup', 'Daily 9:30am', 'Room 402'],
      startX: 40,
      startY: 58,
      lineHeight: 56,
    }),
    path.join(fixturesDir, 'printed-dark-mode.png')
  );
  entries['printed-dark-mode.png'] = {
    filename: 'printed-dark-mode.png',
    groundTruth: 'Team Standup\nDaily 9:30am\nRoom 402',
    expectedStyle: 'printed',
    expectedLineCount: 3,
    tags: ['printed', 'dark-mode'],
    description: 'Dark-mode printed screenshot with >5% bright pixels',
  };

  // 16. Dark-mode sparse (~1.5% bright boundary, matches Finding 5)
  await svgToPng(
    makeTextSvg({
      width: 520,
      height: 180,
      bg: '#111318',
      fg: '#e4e4e7',
      fontFamily: 'Arial',
      fontSize: 22,
      lines: ['Dark Note', 'Check Email'],
      startX: 42,
      startY: 64,
      lineHeight: 46,
    }),
    path.join(fixturesDir, 'printed-dark-mode-sparse.png')
  );
  entries['printed-dark-mode-sparse.png'] = {
    filename: 'printed-dark-mode-sparse.png',
    groundTruth: 'Dark Note\nCheck Email',
    expectedStyle: 'printed',
    expectedLineCount: 2,
    tags: ['printed', 'dark-mode', 'sparse', 'boundary'],
    description: 'Sparse dark-mode screenshot (~1.5% bright pixels boundary)',
  };

  // 17. Dark-mode invoice (Tier 4 S3)
  await svgToPng(
    makeTextSvg({
      width: 680,
      height: 270,
      bg: '#111318',
      fg: '#f4f4f5',
      fontFamily: 'Arial',
      fontSize: 32,
      fontWeight: 'bold',
      lines: [
        'Electric Utility Invoice #8841',
        'Amount Due: $114.50',
        'Due Date: 12/15/2026 at 5:00pm',
        'Pay online urgently',
      ],
      startX: 36,
      startY: 56,
      lineHeight: 56,
    }),
    path.join(fixturesDir, 'invoice-dark-mode.png')
  );
  entries['invoice-dark-mode.png'] = {
    filename: 'invoice-dark-mode.png',
    groundTruth:
      'Electric Utility Invoice #8841\nAmount Due: $114.50\nDue Date: 12/15/2026 at 5:00pm\nPay online urgently',
    expectedStyle: 'printed',
    expectedLineCount: 4,
    tags: ['scenario', 'printed', 'dark-mode', 'invoice'],
    description: 'Dark-mode printed electric utility invoice',
  };

  // 18. Mixed printed header + cursive body (Tier 4 S4)
  const mixedSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="660" height="340">
    <rect width="100%" height="100%" fill="#ffffff" />
    <text x="36" y="64" font-family="Arial" font-size="38" font-weight="bold" fill="#111827">WEEKLY ERRANDS</text>
    <text x="36" y="180" font-family="Segoe Print" font-size="34" fill="#18181b">Pick up prescriptions at 2pm</text>
    <text x="36" y="254" font-family="Segoe Print" font-size="34" fill="#18181b">Buy fresh spinach and salmon</text>
  </svg>`;
  await svgToPng(mixedSvg, path.join(fixturesDir, 'mixed-printed-cursive.png'));
  entries['mixed-printed-cursive.png'] = {
    filename: 'mixed-printed-cursive.png',
    groundTruth:
      'WEEKLY ERRANDS\nPick up prescriptions at 2pm\nBuy fresh spinach and salmon',
    expectedStyle: 'mixed',
    expectedLineCount: 3,
    tags: ['scenario', 'mixed', 'printed', 'cursive'],
    description: 'Mixed printed header and cursive handwritten body',
  };

  // 19. Reminder full flow (Tier 4 S5)
  await svgToPng(
    makeTextSvg({
      width: 880,
      height: 190,
      fontFamily: 'Segoe Print',
      fontSize: 34,
      lines: ['Pay dentist Dr. Lee $85.00 tomorrow 3pm', 'Bring insurance card'],
      startX: 36,
      startY: 66,
      lineHeight: 68,
    }),
    path.join(fixturesDir, 'reminder-full-flow.png')
  );
  entries['reminder-full-flow.png'] = {
    filename: 'reminder-full-flow.png',
    groundTruth: 'Pay dentist Dr. Lee $85.00 tomorrow 3pm\nBring insurance card',
    expectedStyle: 'handwriting',
    expectedLineCount: 2,
    tags: ['scenario', 'reminder'],
    description: 'Handwritten reminder note for full scan -> reminder save flow',
  };

  // 20. Checklist confusions (Tier 4 S6)
  await svgToPng(
    makeTextSvg({
      width: 480,
      height: 410,
      fontFamily: 'Segoe Print',
      fontSize: 38,
      lines: ['morning', 'milk', 'date', 'work', 'spinach', 'dinner'],
      startX: 44,
      startY: 56,
      lineHeight: 62,
    }),
    path.join(fixturesDir, 'checklist-confusions.png')
  );
  entries['checklist-confusions.png'] = {
    filename: 'checklist-confusions.png',
    groundTruth: 'morning\nmilk\ndate\nwork\nspinach\ndinner',
    expectedStyle: 'handwriting',
    expectedLineCount: 6,
    tags: ['scenario', 'cursive', 'checklist', 'confusions'],
    description: '6-line handwritten checklist with confusion-prone words',
  };

  // 21. Zero-margin tight crop (small text touching edges)
  await svgToPng(
    makeTextSvg({
      width: 170,
      height: 36,
      fontFamily: 'Arial',
      fontSize: 16,
      fontWeight: 'bold',
      lines: ['Buy Milk Today'],
      startX: 1,
      startY: 22,
    }),
    path.join(fixturesDir, 'zero-margin-crop.png')
  );
  entries['zero-margin-crop.png'] = {
    filename: 'zero-margin-crop.png',
    groundTruth: 'Buy Milk Today',
    expectedStyle: 'printed',
    expectedLineCount: 1,
    tags: ['boundary', 'zero-margin', 'dpi-upscale'],
    description: 'Zero-margin tightly cropped small text',
  };

  // 22. Blank white
  await sharp({
    create: { width: 400, height: 200, channels: 3, background: { r: 255, g: 255, b: 255 } },
  })
    .png()
    .toFile(path.join(fixturesDir, 'blank-white.png'));
  entries['blank-white.png'] = {
    filename: 'blank-white.png',
    groundTruth: '',
    expectedStyle: 'blank',
    expectedLineCount: 0,
    tags: ['boundary', 'blank'],
    description: '400x200 blank white image',
  };

  // 23. Solid black
  await sharp({
    create: { width: 200, height: 200, channels: 3, background: { r: 0, g: 0, b: 0 } },
  })
    .png()
    .toFile(path.join(fixturesDir, 'solid-black.png'));
  entries['solid-black.png'] = {
    filename: 'solid-black.png',
    groundTruth: '',
    expectedStyle: 'blank',
    expectedLineCount: 0,
    tags: ['boundary', 'black'],
    description: '200x200 solid black image',
  };

  // 24. Tiny 1x1
  await sharp({
    create: { width: 1, height: 1, channels: 3, background: { r: 255, g: 255, b: 255 } },
  })
    .png()
    .toFile(path.join(fixturesDir, 'tiny-1x1.png'));
  entries['tiny-1x1.png'] = {
    filename: 'tiny-1x1.png',
    groundTruth: '',
    expectedStyle: 'blank',
    expectedLineCount: 0,
    tags: ['boundary', 'tiny'],
    description: '1x1 pixel PNG image',
  };

  // 25. Noisy paper (deterministic pseudo-random speckles, no words)
  const nw = 320;
  const nh = 160;
  const noiseBuf = Buffer.alloc(nw * nh * 3, 245);
  let seed = 1337;
  for (let i = 0; i < nw * nh; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const r = seed % 100;
    const val = r < 4 ? 120 : r < 12 ? 205 : 244;
    noiseBuf[i * 3] = val;
    noiseBuf[i * 3 + 1] = val;
    noiseBuf[i * 3 + 2] = val;
  }
  await sharp(noiseBuf, { raw: { width: nw, height: nh, channels: 3 } })
    .png()
    .toFile(path.join(fixturesDir, 'noisy-paper.png'));
  entries['noisy-paper.png'] = {
    filename: 'noisy-paper.png',
    groundTruth: '',
    expectedStyle: 'noise',
    expectedLineCount: 0,
    tags: ['boundary', 'noise', 'hallucination'],
    description: 'Textured noisy paper without words',
  };

  // 26. Non-grocery 7 lines (tests removal of canonicalGroceryItems >= 6 lines mock)
  await svgToPng(
    makeTextSvg({
      width: 440,
      height: 460,
      fontFamily: 'Segoe Print',
      fontSize: 36,
      lines: ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf'],
      startX: 44,
      startY: 56,
      lineHeight: 58,
    }),
    path.join(fixturesDir, 'non-grocery-7lines.png')
  );
  entries['non-grocery-7lines.png'] = {
    filename: 'non-grocery-7lines.png',
    groundTruth: 'alpha\nbravo\ncharlie\ndelta\necho\nfoxtrot\ngolf',
    expectedStyle: 'handwriting',
    expectedLineCount: 7,
    tags: ['cursive', 'non-grocery', 'no-mock'],
    description: '7-line non-grocery handwriting note verifying no hardcoded grocery fallback',
  };

  // 27. 15-line printed note (for UI scrolling confidence list)
  const lines15 = Array.from({ length: 15 }, (_, i) => `Task ${String(i + 1).padStart(2, '0')} ready`);
  await svgToPng(
    makeTextSvg({
      width: 480,
      height: 680,
      fontFamily: 'Arial',
      fontSize: 26,
      fontWeight: '600',
      lines: lines15,
      startX: 32,
      startY: 42,
      lineHeight: 42,
    }),
    path.join(fixturesDir, 'printed-15lines.png')
  );
  entries['printed-15lines.png'] = {
    filename: 'printed-15lines.png',
    groundTruth: lines15.join('\n'),
    expectedStyle: 'printed',
    expectedLineCount: 15,
    tags: ['printed', '15lines', 'ui'],
    description: '15-line printed task list for UI scrolling line-confidence check',
  };

  // 28. Single-line cursive
  await svgToPng(
    makeTextSvg({
      width: 460,
      height: 110,
      fontFamily: 'Segoe Print',
      fontSize: 40,
      lines: ['buy milk today'],
      startX: 36,
      startY: 70,
    }),
    path.join(fixturesDir, 'cursive-1line.png')
  );
  entries['cursive-1line.png'] = {
    filename: 'cursive-1line.png',
    groundTruth: 'buy milk today',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['cursive', '1line'],
    description: 'Single-line cursive note',
  };

  // 29. High-res 3200x2400 printed PNG
  await svgToPng(
    makeTextSvg({
      width: 3200,
      height: 2400,
      fontFamily: 'Arial',
      fontSize: 120,
      fontWeight: 'bold',
      lines: ['High Resolution Scan', 'Check Schedule 4pm'],
      startX: 220,
      startY: 600,
      lineHeight: 220,
    }),
    path.join(fixturesDir, 'highres-3200x2400.png')
  );
  entries['highres-3200x2400.png'] = {
    filename: 'highres-3200x2400.png',
    groundTruth: 'High Resolution Scan\nCheck Schedule 4pm',
    expectedStyle: 'printed',
    expectedLineCount: 2,
    tags: ['boundary', 'highres'],
    description: 'High-resolution 3200x2400 PNG testing server downscale handling',
  };

  // 30. Single word "milk"
  await svgToPng(
    makeTextSvg({
      width: 240,
      height: 110,
      fontFamily: 'Segoe Print',
      fontSize: 44,
      lines: ['milk'],
      startX: 36,
      startY: 70,
    }),
    path.join(fixturesDir, 'single-word-milk.png')
  );
  entries['single-word-milk.png'] = {
    filename: 'single-word-milk.png',
    groundTruth: 'milk',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['boundary', 'single-word'],
    description: 'Single-word handwritten crop',
  };

  // 31. Single char "7"
  await svgToPng(
    makeTextSvg({
      width: 120,
      height: 120,
      fontFamily: 'Segoe Print',
      fontSize: 52,
      lines: ['7'],
      startX: 36,
      startY: 80,
    }),
    path.join(fixturesDir, 'single-char-7.png')
  );
  entries['single-char-7.png'] = {
    filename: 'single-char-7.png',
    groundTruth: '7',
    expectedStyle: 'handwriting',
    expectedLineCount: 1,
    tags: ['boundary', 'single-char'],
    description: 'Single-character handwritten crop',
  };

  const manifest: Manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    fixtures: entries,
  };
  fs.writeFileSync(
    path.join(fixturesDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
    'utf8'
  );
  return manifest;
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))
) {
  const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'fixtures');
  generateAllFixtures(dir)
    .then((m) => {
      console.log(`Generated ${Object.keys(m.fixtures).length} fixtures in ${dir}`);
    })
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
