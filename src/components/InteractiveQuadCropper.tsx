import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Crop,
  RefreshCw,
  Sparkles,
  Check,
  Loader2,
  RotateCcw,
  RotateCw,
  AlertTriangle,
  FileText,
  Eye,
  EyeOff,
  Layers,
  Sparkle,
} from 'lucide-react';
import {
  Point2D,
  QuadCorners,
  orderQuadCorners,
} from '../lib/ocr/geometry';
import {
  detectDocumentPerimeter,
  dewarpImageWithQuad,
  detectTextLinesAndHull,
  DetectedTextLine,
  NormalizedQuad,
  isBoxInPolygon,
  rotateImageBase64,
} from '../lib/ocr/documentEdgeDetector';

interface InteractiveQuadCropperProps {
  imageSource: string;
  onApplyDewarp: (dewarpedDataUrl: string) => void;
  onCancelOrFull: () => void;
  onRetake?: () => void;
}

type AspectPreset = 'free' | 'a4' | 'letter' | 'square' | '4:3';

export const InteractiveQuadCropper: React.FC<InteractiveQuadCropperProps> = ({
  imageSource,
  onApplyDewarp,
  onCancelOrFull,
  onRetake,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const loupeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Active image source (supports 90° rotation directly in cropper)
  const [currentImage, setCurrentImage] = useState<string>(imageSource);

  const [naturalWidth, setNaturalWidth] = useState(0);
  const [naturalHeight, setNaturalHeight] = useState(0);
  const [displayedWidth, setDisplayedWidth] = useState(0);
  const [displayedHeight, setDisplayedHeight] = useState(0);
  const [imageOffsetLeft, setImageOffsetLeft] = useState(0);
  const [imageOffsetTop, setImageOffsetTop] = useState(0);

  // Normalized corner coordinates (0..1)
  const [corners, setCorners] = useState<[Point2D, Point2D, Point2D, Point2D]>([
    { x: 0.05, y: 0.05 },
    { x: 0.95, y: 0.05 },
    { x: 0.95, y: 0.95 },
    { x: 0.05, y: 0.95 },
  ]);

  const [activeCornerIdx, setActiveCornerIdx] = useState<number | null>(null);
  const [isDetecting, setIsDetecting] = useState(true);
  const [isDewarping, setIsDewarping] = useState(false);
  const [isRotating, setIsRotating] = useState(false);

  // Document perimeter detection result
  const [sheetQuad, setSheetQuad] = useState<NormalizedQuad | null>(null);
  const [autoDetectedSheet, setAutoDetectedSheet] = useState(false);

  // Text-first detection state
  const [detectedLines, setDetectedLines] = useState<DetectedTextLine[]>([]);
  const [textHullQuad, setTextHullQuad] = useState<NormalizedQuad | null>(null);
  const [showTextBoxes, setShowTextBoxes] = useState(true);

  // Enhancement & Aspect controls
  const [enhanceMode, setEnhanceMode] = useState<'clean' | 'natural'>('clean');
  const [aspectPreset, setAspectPreset] = useState<AspectPreset>('free');

  // Loupe position state
  const [loupePoint, setLoupePoint] = useState<Point2D | null>(null);

  // Update currentImage whenever prop changes
  useEffect(() => {
    setCurrentImage(imageSource);
  }, [imageSource]);

  // Measure rendered image rect inside container
  const updateLayoutMetrics = useCallback(() => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img || !img.naturalWidth || !img.naturalHeight) return;

    const cW = container.clientWidth;
    const cH = container.clientHeight;
    const nW = img.naturalWidth;
    const nH = img.naturalHeight;

    const imgAspect = nW / nH;
    const containerAspect = cW / cH;

    let dW = cW;
    let dH = cH;
    let offX = 0;
    let offY = 0;

    if (imgAspect > containerAspect) {
      dW = cW;
      dH = Math.round(cW / imgAspect);
      offX = 0;
      offY = Math.round((cH - dH) / 2);
    } else {
      dH = cH;
      dW = Math.round(cH * imgAspect);
      offX = Math.round((cW - dW) / 2);
      offY = 0;
    }

    setNaturalWidth(nW);
    setNaturalHeight(nH);
    setDisplayedWidth(dW);
    setDisplayedHeight(dH);
    setImageOffsetLeft(offX);
    setImageOffsetTop(offY);
  }, []);

  // Run Text-First visual detection & document perimeter analysis
  const runDetection = useCallback(async (sourceImg: string) => {
    setIsDetecting(true);

    try {
      // 1. Detect text lines & hull first
      const textResult = await detectTextLinesAndHull(sourceImg);
      setDetectedLines(textResult.lines);
      setTextHullQuad(textResult.normalizedTextHull);

      // 2. Detect sheet perimeter
      const perimeterResult = await detectDocumentPerimeter(sourceImg);
      setSheetQuad(perimeterResult.normalizedQuad);
      setAutoDetectedSheet(perimeterResult.autoDetected);

      // Auto-frame decision: prioritize text hull if text was detected,
      // otherwise use document sheet perimeter, fallback to inset
      if (textResult.normalizedTextHull && textResult.lines.length > 0) {
        setCorners([
          { x: textResult.normalizedTextHull[0].x, y: textResult.normalizedTextHull[0].y },
          { x: textResult.normalizedTextHull[1].x, y: textResult.normalizedTextHull[1].y },
          { x: textResult.normalizedTextHull[2].x, y: textResult.normalizedTextHull[2].y },
          { x: textResult.normalizedTextHull[3].x, y: textResult.normalizedTextHull[3].y },
        ]);
      } else if (perimeterResult.autoDetected) {
        setCorners([
          { x: perimeterResult.normalizedQuad[0].x, y: perimeterResult.normalizedQuad[0].y },
          { x: perimeterResult.normalizedQuad[1].x, y: perimeterResult.normalizedQuad[1].y },
          { x: perimeterResult.normalizedQuad[2].x, y: perimeterResult.normalizedQuad[2].y },
          { x: perimeterResult.normalizedQuad[3].x, y: perimeterResult.normalizedQuad[3].y },
        ]);
      } else {
        setCorners([
          { x: 0.05, y: 0.05 },
          { x: 0.95, y: 0.05 },
          { x: 0.95, y: 0.95 },
          { x: 0.05, y: 0.95 },
        ]);
      }
    } catch (err) {
      console.warn('In-cropper text & perimeter detection fallback:', err);
      setCorners([
        { x: 0.05, y: 0.05 },
        { x: 0.95, y: 0.05 },
        { x: 0.95, y: 0.95 },
        { x: 0.05, y: 0.95 },
      ]);
    } finally {
      setIsDetecting(false);
    }
  }, []);

  useEffect(() => {
    runDetection(currentImage);
  }, [currentImage, runDetection]);

  useEffect(() => {
    window.addEventListener('resize', updateLayoutMetrics);
    return () => window.removeEventListener('resize', updateLayoutMetrics);
  }, [updateLayoutMetrics]);

  // Real-time ray-casting containment check: identify any text lines outside the crop quad
  const clippedLineIndices = useMemo(() => {
    if (detectedLines.length === 0) return [];
    const clipped: number[] = [];
    detectedLines.forEach((line, idx) => {
      const isInside = isBoxInPolygon(line.normalized, corners);
      if (!isInside) {
        clipped.push(idx);
      }
    });
    return clipped;
  }, [detectedLines, corners]);

  // Update floating loupe canvas during corner dragging
  const updateLoupeCanvas = useCallback(
    (normX: number, normY: number) => {
      const img = imgRef.current;
      const canvas = loupeCanvasRef.current;
      if (!img || !canvas || !img.naturalWidth || !img.naturalHeight) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const pxX = normX * img.naturalWidth;
      const pxY = normY * img.naturalHeight;

      const cropSize = 90; // window in source pixels to zoom
      const srcX = pxX - cropSize / 2;
      const srcY = pxY - cropSize / 2;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(
        img,
        srcX,
        srcY,
        cropSize,
        cropSize,
        0,
        0,
        canvas.width,
        canvas.height
      );

      // Draw subtle reticle crosshairs in the loupe
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      ctx.strokeStyle = '#22d3ee'; // cyan-400
      ctx.lineWidth = 1.5;

      ctx.beginPath();
      ctx.moveTo(cx - 16, cy);
      ctx.lineTo(cx - 4, cy);
      ctx.moveTo(cx + 4, cy);
      ctx.lineTo(cx + 16, cy);
      ctx.moveTo(cx, cy - 16);
      ctx.lineTo(cx, cy - 4);
      ctx.moveTo(cx, cy + 4);
      ctx.lineTo(cx, cy + 16);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#22d3ee';
      ctx.fill();
    },
    []
  );

  // Magnetic edge snapping helper: snaps coordinate if within snap threshold of borders or text hull
  const applyMagneticSnap = useCallback(
    (val: number, snapTargets: number[], threshold = 0.03): number => {
      for (const target of snapTargets) {
        if (Math.abs(val - target) <= threshold) {
          return target;
        }
      }
      return val;
    },
    []
  );

  // Pointer drag event handlers for corner pins
  const handlePointerDown = (idx: number, e: React.PointerEvent<HTMLButtonElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setActiveCornerIdx(idx);

    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      setLoupePoint({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    }
    updateLoupeCanvas(corners[idx].x, corners[idx].y);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (activeCornerIdx === null || displayedWidth === 0 || displayedHeight === 0) return;
    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const relX = e.clientX - rect.left - imageOffsetLeft;
    const relY = e.clientY - rect.top - imageOffsetTop;

    let clampedX = Math.max(0, Math.min(1, relX / displayedWidth));
    let clampedY = Math.max(0, Math.min(1, relY / displayedHeight));

    // Magnetic edge snapping targets: image borders (0, 1) and text hull edges if present
    const snapTargetsX = [0, 1];
    const snapTargetsY = [0, 1];
    if (textHullQuad) {
      snapTargetsX.push(textHullQuad[0].x, textHullQuad[1].x);
      snapTargetsY.push(textHullQuad[0].y, textHullQuad[2].y);
    }

    clampedX = applyMagneticSnap(clampedX, snapTargetsX, 0.025);
    clampedY = applyMagneticSnap(clampedY, snapTargetsY, 0.025);

    setCorners((prev) => {
      const next = [...prev] as [Point2D, Point2D, Point2D, Point2D];
      next[activeCornerIdx] = { x: clampedX, y: clampedY };
      return next;
    });

    setAspectPreset('free');

    setLoupePoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    updateLoupeCanvas(clampedX, clampedY);
  };

  const handlePointerUp = () => {
    if (activeCornerIdx !== null) {
      setActiveCornerIdx(null);
      setLoupePoint(null);
    }
  };

  // One-tap: Snap to Text
  const handleSnapToText = () => {
    if (textHullQuad) {
      setCorners([
        { x: textHullQuad[0].x, y: textHullQuad[0].y },
        { x: textHullQuad[1].x, y: textHullQuad[1].y },
        { x: textHullQuad[2].x, y: textHullQuad[2].y },
        { x: textHullQuad[3].x, y: textHullQuad[3].y },
      ]);
      setAspectPreset('free');
    }
  };

  // One-tap: Snap to Sheet
  const handleSnapToSheet = () => {
    if (sheetQuad) {
      setCorners([
        { x: sheetQuad[0].x, y: sheetQuad[0].y },
        { x: sheetQuad[1].x, y: sheetQuad[1].y },
        { x: sheetQuad[2].x, y: sheetQuad[2].y },
        { x: sheetQuad[3].x, y: sheetQuad[3].y },
      ]);
    } else {
      setCorners([
        { x: 0.04, y: 0.04 },
        { x: 0.96, y: 0.04 },
        { x: 0.96, y: 0.96 },
        { x: 0.04, y: 0.96 },
      ]);
    }
    setAspectPreset('free');
  };

  // Aspect ratio preset setter
  const handleSelectAspect = (preset: AspectPreset) => {
    setAspectPreset(preset);
    if (preset === 'free' || naturalWidth === 0 || naturalHeight === 0) return;

    let targetRatio = 1; // W / H
    if (preset === 'a4') targetRatio = 1 / 1.414;
    else if (preset === 'letter') targetRatio = 8.5 / 11;
    else if (preset === 'square') targetRatio = 1.0;
    else if (preset === '4:3') targetRatio = 4 / 3;

    // Center the aspect preset around current corners or text hull
    let centerX = 0.5;
    let centerY = 0.5;
    if (textHullQuad) {
      centerX = (textHullQuad[0].x + textHullQuad[1].x) / 2;
      centerY = (textHullQuad[0].y + textHullQuad[2].y) / 2;
    }

    const normAspect = targetRatio * (naturalHeight / naturalWidth);
    let wNorm = 0.88;
    let hNorm = wNorm / normAspect;
    if (hNorm > 0.88) {
      hNorm = 0.88;
      wNorm = hNorm * normAspect;
    }

    const x0 = Math.max(0, Math.min(1 - wNorm, centerX - wNorm / 2));
    const y0 = Math.max(0, Math.min(1 - hNorm, centerY - hNorm / 2));
    const x1 = Math.min(1, x0 + wNorm);
    const y1 = Math.min(1, y0 + hNorm);

    setCorners([
      { x: x0, y: y0 },
      { x: x1, y: y0 },
      { x: x1, y: y1 },
      { x: x0, y: y1 },
    ]);
  };

  // 90° Quick Rotation
  const handleRotate = async (deltaDeg: number) => {
    setIsRotating(true);
    try {
      const rotated = await rotateImageBase64(currentImage, deltaDeg);
      setCurrentImage(rotated);
      // Re-run detection on the rotated canvas
      await runDetection(rotated);
    } catch (err) {
      console.error('Failed to rotate image:', err);
    } finally {
      setIsRotating(false);
    }
  };

  // Execute modern OCR dewarp with 300 DPI equivalent resampling and Retinex doc cleaning
  const handleExecuteDewarp = async () => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) {
      onCancelOrFull();
      return;
    }

    setIsDewarping(true);
    try {
      const nW = img.naturalWidth;
      const nH = img.naturalHeight;
      const pixelCorners: QuadCorners = [
        { x: corners[0].x * nW, y: corners[0].y * nH },
        { x: corners[1].x * nW, y: corners[1].y * nH },
        { x: corners[2].x * nW, y: corners[2].y * nH },
        { x: corners[3].x * nW, y: corners[3].y * nH },
      ];

      const ordered = orderQuadCorners(pixelCorners);
      const res = await dewarpImageWithQuad(currentImage, ordered, {
        maxWidth: 2200, // 300 DPI equivalent standard
        maxHeight: 2200,
        enhanceDocLighting: enhanceMode === 'clean',
      });

      onApplyDewarp(res.dataUrl);
    } catch (err) {
      console.error('Dewarp failed, proceeding with original:', err);
      onCancelOrFull();
    } finally {
      setIsDewarping(false);
    }
  };

  // SVG polygon points in container pixels
  const svgPoints = corners
    .map((c) => {
      const px = imageOffsetLeft + c.x * displayedWidth;
      const py = imageOffsetTop + c.y * displayedHeight;
      return `${px},${py}`;
    })
    .join(' ');

  const cornerLabels = ['Top Left', 'Top Right', 'Bottom Right', 'Bottom Left'];
  const hasDetectedText = detectedLines.length > 0;
  const hasClippedLines = clippedLineIndices.length > 0;

  return (
    <div className="space-y-3 select-none">
      {/* Real-time Text Framing & Containment Status Banner */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-2">
          {hasClippedLines ? (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/15 border border-amber-500/40 text-amber-600 dark:text-amber-400 text-xs font-semibold animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              <span>
                {clippedLineIndices.length} text line{clippedLineIndices.length > 1 ? 's' : ''} outside crop!
              </span>
            </div>
          ) : hasDetectedText ? (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/15 border border-cyan-500/40 text-cyan-700 dark:text-cyan-300 text-xs font-medium">
              <Sparkles className="w-3.5 h-3.5 shrink-0 text-cyan-500" />
              <span>
                {detectedLines.length} text line{detectedLines.length > 1 ? 's' : ''} detected · Frame aligned
              </span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 text-xs font-medium">
              <Crop className="w-3.5 h-3.5 shrink-0" />
              <span>Adjust 4 corners to frame note</span>
            </div>
          )}

          <span className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 hidden sm:inline">
            300 DPI Std
          </span>
        </div>

        {/* Rotation & View toggles */}
        <div className="flex items-center gap-1">
          {hasDetectedText && (
            <button
              type="button"
              onClick={() => setShowTextBoxes((prev) => !prev)}
              className={`min-h-[36px] min-w-[36px] px-2 rounded-lg border text-xs font-medium inline-flex items-center gap-1 transition-colors ${
                showTextBoxes
                  ? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-600 dark:text-cyan-300'
                  : 'border-zinc-300 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400'
              }`}
              title={showTextBoxes ? 'Hide text boxes overlay' : 'Show detected text boxes'}
              aria-label="Toggle text boxes"
            >
              {showTextBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">Text</span>
            </button>
          )}

          <button
            type="button"
            disabled={isRotating || isDewarping}
            onClick={() => handleRotate(-90)}
            className="min-h-[36px] min-w-[36px] px-2 rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 active:scale-95 text-zinc-700 dark:text-zinc-200 text-xs font-medium inline-flex items-center justify-center transition-transform"
            title="Rotate 90° counter-clockwise"
            aria-label="Rotate counter-clockwise"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            disabled={isRotating || isDewarping}
            onClick={() => handleRotate(90)}
            className="min-h-[36px] min-w-[36px] px-2 rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 active:scale-95 text-zinc-700 dark:text-zinc-200 text-xs font-medium inline-flex items-center justify-center transition-transform"
            title="Rotate 90° clockwise"
            aria-label="Rotate clockwise"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Interactive Dewarp Canvas Surface */}
      <div
        ref={containerRef}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        className="relative rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-950 aspect-[3/4] sm:aspect-[4/3] max-h-[52vh] min-h-[300px] w-full flex items-center justify-center touch-none shadow-inner"
      >
        <img
          ref={imgRef}
          src={currentImage}
          alt="Document to crop and dewarp"
          onLoad={updateLayoutMetrics}
          className="w-full h-full object-contain pointer-events-none select-none"
        />

        {/* Loading overlay for detection / rotation / dewarping */}
        {(isDetecting || isDewarping || isRotating) && (
          <div className="absolute inset-0 bg-black/70 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2.5 text-zinc-200 text-xs p-4 text-center z-30">
            <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
            <span className="font-semibold text-sm">
              {isRotating
                ? 'Rotating 90° & re-analyzing...'
                : isDewarping
                ? 'Dewarping & flattening document...'
                : 'Detecting text lines on device...'}
            </span>
            <span className="text-[11px] text-zinc-400">
              {isRotating
                ? 'Recomputing text boundaries'
                : isDewarping
                ? 'Applying 300 DPI Retinex illumination filter'
                : 'Tracing words and paragraph contours'}
            </span>
          </div>
        )}

        {/* Text-First Visual Glowing Ribbons & Crop Scrim SVG Layer */}
        {displayedWidth > 0 && displayedHeight > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
            {/* Shaded mask over the exterior */}
            <defs>
              <mask id="quad-crop-mask">
                <rect width="100%" height="100%" fill="white" />
                <polygon points={svgPoints} fill="black" />
              </mask>
              <filter id="text-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Dark scrim outside the crop quadrilateral */}
            <rect
              width="100%"
              height="100%"
              fill="rgba(0, 0, 0, 0.45)"
              mask="url(#quad-crop-mask)"
            />

            {/* Glowing Text Boxes Highlight Layer */}
            {showTextBoxes &&
              detectedLines.map((line, idx) => {
                const isClipped = clippedLineIndices.includes(idx);
                const rx = imageOffsetLeft + line.normalized.x * displayedWidth;
                const ry = imageOffsetTop + line.normalized.y * displayedHeight;
                const rw = line.normalized.width * displayedWidth;
                const rh = line.normalized.height * displayedHeight;

                return (
                  <g key={line.id}>
                    <rect
                      x={rx}
                      y={ry}
                      width={rw}
                      height={rh}
                      rx={3}
                      ry={3}
                      fill={isClipped ? 'rgba(245, 158, 11, 0.28)' : 'rgba(34, 211, 238, 0.18)'}
                      stroke={isClipped ? '#f59e0b' : '#22d3ee'}
                      strokeWidth={isClipped ? 1.8 : 1.2}
                      strokeDasharray={isClipped ? '4 2' : undefined}
                      className="transition-colors duration-200"
                    />
                    {isClipped && (
                      <circle
                        cx={rx + rw}
                        cy={ry}
                        r={4.5}
                        fill="#f59e0b"
                        stroke="#ffffff"
                        strokeWidth={1}
                      />
                    )}
                  </g>
                );
              })}

            {/* Active Crop Polygon Boundary */}
            <polygon
              points={svgPoints}
              className={`fill-cyan-500/10 stroke-2 transition-colors ${
                hasClippedLines ? 'stroke-amber-400' : 'stroke-cyan-400'
              }`}
              strokeDasharray={hasClippedLines ? '6 3' : '4 2'}
            />
          </svg>
        )}

        {/* 4 Interactive Corner Pins (Touch targets >= 48px) */}
        {displayedWidth > 0 &&
          displayedHeight > 0 &&
          corners.map((c, idx) => {
            const posX = imageOffsetLeft + c.x * displayedWidth;
            const posY = imageOffsetTop + c.y * displayedHeight;
            const isActive = activeCornerIdx === idx;

            return (
              <button
                key={idx}
                type="button"
                onPointerDown={(e) => handlePointerDown(idx, e)}
                style={{
                  left: `${posX}px`,
                  top: `${posY}px`,
                }}
                className="absolute -translate-x-1/2 -translate-y-1/2 min-h-[48px] min-w-[48px] flex items-center justify-center cursor-grab active:cursor-grabbing z-20 touch-none group"
                aria-label={`Drag ${cornerLabels[idx]} corner`}
              >
                {/* Visual pin target */}
                <div
                  className={`w-5 h-5 rounded-full border-2 transition-transform ${
                    isActive
                      ? hasClippedLines
                        ? 'scale-125 bg-white border-amber-400 ring-4 ring-amber-400/50 shadow-lg'
                        : 'scale-125 bg-white border-cyan-400 ring-4 ring-cyan-400/50 shadow-lg'
                      : hasClippedLines
                      ? 'bg-white border-amber-500 shadow-md group-hover:scale-110'
                      : 'bg-white border-cyan-500 shadow-md group-hover:scale-110'
                  }`}
                >
                  <div
                    className={`w-1.5 h-1.5 rounded-full m-auto mt-1 ${
                      hasClippedLines ? 'bg-amber-600' : 'bg-cyan-600'
                    }`}
                  />
                </div>
              </button>
            );
          })}

        {/* Floating 2.5x Magnification Loupe HUD */}
        {activeCornerIdx !== null && loupePoint && (
          <div
            style={{
              left: `${Math.max(48, Math.min(displayedWidth + imageOffsetLeft - 48, loupePoint.x))}px`,
              top: `${Math.max(56, loupePoint.y - 75)}px`,
            }}
            className="absolute -translate-x-1/2 -translate-y-1/2 z-40 pointer-events-none flex flex-col items-center"
          >
            <div className="w-20 h-20 rounded-full border-2 border-cyan-300 shadow-2xl overflow-hidden bg-black/90 ring-4 ring-cyan-500/30">
              <canvas
                ref={loupeCanvasRef}
                width={80}
                height={80}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="w-2 h-2 rotate-45 bg-cyan-400 -mt-1 shadow-xs" />
          </div>
        )}
      </div>

      {/* Aspect Ratio & Auto-Snap Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-zinc-200 dark:border-zinc-800">
        {/* Snap Buttons */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <button
            type="button"
            disabled={!textHullQuad || detectedLines.length === 0}
            onClick={handleSnapToText}
            className="min-h-[44px] px-3 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 active:scale-95 disabled:opacity-40 text-cyan-600 dark:text-cyan-300 border border-cyan-500/30 text-xs font-semibold inline-flex items-center gap-1.5 whitespace-nowrap transition-all"
            title="Snap 4 corners tightly around all detected text lines"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Snap to Text ({detectedLines.length})</span>
          </button>

          <button
            type="button"
            onClick={handleSnapToSheet}
            className="min-h-[44px] px-3 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 active:scale-95 text-zinc-700 dark:text-zinc-200 border border-zinc-200 dark:border-zinc-700 text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap transition-all"
            title="Snap to the outer document / paper sheet perimeter"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Snap to Sheet</span>
          </button>
        </div>

        {/* Aspect Ratio Presets */}
        <div className="flex items-center gap-1 bg-zinc-100 dark:bg-zinc-800/80 p-0.5 rounded-xl border border-zinc-200 dark:border-zinc-700">
          {(['free', 'a4', 'letter', 'square'] as AspectPreset[]).map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => handleSelectAspect(preset)}
              className={`min-h-[38px] px-2.5 rounded-lg text-xs font-medium capitalize transition-all ${
                aspectPreset === preset
                  ? 'bg-white dark:bg-zinc-900 text-cyan-600 dark:text-cyan-400 shadow-xs font-semibold'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              {preset === 'a4' ? 'A4' : preset === 'letter' ? 'Letter' : preset}
            </button>
          ))}
        </div>
      </div>

      {/* Document Enhancement Filter & Primary Actions */}
      <div className="space-y-2.5 pt-1">
        {/* Filter Selection Segmented Control */}
        <div className="flex items-center justify-between gap-2 p-1.5 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-xs">
          <span className="text-zinc-500 dark:text-zinc-400 font-medium px-1 flex items-center gap-1.5">
            <Sparkle className="w-3.5 h-3.5 text-cyan-500" />
            <span>Output Filter:</span>
          </span>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setEnhanceMode('clean')}
              className={`min-h-[38px] px-3 rounded-lg text-xs font-medium transition-all ${
                enhanceMode === 'clean'
                  ? 'bg-cyan-600 text-white shadow-xs font-semibold'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              Doc Clean (Flatten & Boost)
            </button>
            <button
              type="button"
              onClick={() => setEnhanceMode('natural')}
              className={`min-h-[38px] px-3 rounded-lg text-xs font-medium transition-all ${
                enhanceMode === 'natural'
                  ? 'bg-zinc-700 text-white shadow-xs font-semibold'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              Natural Photo
            </button>
          </div>
        </div>

        {/* Primary Action Buttons (Mobile >= 48px hitboxes) */}
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            disabled={isDewarping || isDetecting || isRotating}
            onClick={handleExecuteDewarp}
            className={`min-h-[48px] flex-1 px-4 rounded-xl active:scale-[0.98] disabled:opacity-50 text-white text-sm font-semibold inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all shadow-sm ${
              hasClippedLines
                ? 'bg-amber-600 hover:bg-amber-700 ring-2 ring-amber-400/40'
                : 'bg-cyan-600 hover:bg-cyan-700'
            }`}
          >
            {isDewarping ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : hasClippedLines ? (
              <AlertTriangle className="w-4 h-4" />
            ) : (
              <Check className="w-4 h-4" />
            )}
            <span>
              {hasClippedLines
                ? `Crop Anyway (${clippedLineIndices.length} line outside)`
                : 'Dewarp & Read Note (300 DPI)'}
            </span>
          </button>

          <button
            type="button"
            disabled={isDewarping}
            onClick={onCancelOrFull}
            className="min-h-[48px] px-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <span>Skip Dewarp</span>
          </button>

          {onRetake && (
            <button
              type="button"
              disabled={isDewarping}
              onClick={onRetake}
              className="min-h-[48px] px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1 whitespace-nowrap shrink-0"
              title="Retake photo"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Retake</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
