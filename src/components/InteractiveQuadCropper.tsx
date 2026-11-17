import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Crop,
  RefreshCw,
  Sparkles,
  Maximize2,
  Check,
  X,
  Loader2,
  Move,
} from 'lucide-react';
import {
  Point2D,
  QuadCorners,
  orderQuadCorners,
} from '../lib/ocr/geometry';
import {
  detectDocumentPerimeter,
  dewarpImageWithQuad,
} from '../lib/ocr/documentEdgeDetector';

interface InteractiveQuadCropperProps {
  imageSource: string;
  onApplyDewarp: (dewarpedDataUrl: string) => void;
  onCancelOrFull: () => void;
  onRetake?: () => void;
}

export const InteractiveQuadCropper: React.FC<InteractiveQuadCropperProps> = ({
  imageSource,
  onApplyDewarp,
  onCancelOrFull,
  onRetake,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const loupeCanvasRef = useRef<HTMLCanvasElement | null>(null);

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
  const [autoDetected, setAutoDetected] = useState(false);
  const [detectedMode, setDetectedMode] = useState<'sheet' | 'text-block' | 'inset'>('inset');

  // Loupe position state
  const [loupePoint, setLoupePoint] = useState<Point2D | null>(null);

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

  // Run on-device perimeter detection on mount
  useEffect(() => {
    let active = true;
    setIsDetecting(true);

    detectDocumentPerimeter(imageSource)
      .then((res) => {
        if (!active) return;
        setCorners([
          { x: res.normalizedQuad[0].x, y: res.normalizedQuad[0].y },
          { x: res.normalizedQuad[1].x, y: res.normalizedQuad[1].y },
          { x: res.normalizedQuad[2].x, y: res.normalizedQuad[2].y },
          { x: res.normalizedQuad[3].x, y: res.normalizedQuad[3].y },
        ]);
        setAutoDetected(res.autoDetected);
        setDetectedMode(res.mode);
      })
      .catch((err) => {
        console.warn('Auto perimeter detection fallback:', err);
      })
      .finally(() => {
        if (active) setIsDetecting(false);
      });

    return () => {
      active = false;
    };
  }, [imageSource]);

  useEffect(() => {
    window.addEventListener('resize', updateLayoutMetrics);
    return () => window.removeEventListener('resize', updateLayoutMetrics);
  }, [updateLayoutMetrics]);

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

    const clampedX = Math.max(0, Math.min(1, relX / displayedWidth));
    const clampedY = Math.max(0, Math.min(1, relY / displayedHeight));

    setCorners((prev) => {
      const next = [...prev] as [Point2D, Point2D, Point2D, Point2D];
      next[activeCornerIdx] = { x: clampedX, y: clampedY };
      return next;
    });

    setLoupePoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    updateLoupeCanvas(clampedX, clampedY);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (activeCornerIdx !== null) {
      setActiveCornerIdx(null);
      setLoupePoint(null);
    }
  };

  const handleResetToInset = () => {
    setCorners([
      { x: 0.04, y: 0.04 },
      { x: 0.96, y: 0.04 },
      { x: 0.96, y: 0.96 },
      { x: 0.04, y: 0.96 },
    ]);
  };

  const handleReDetect = async () => {
    setIsDetecting(true);
    try {
      const res = await detectDocumentPerimeter(imageSource);
      setCorners([
        { x: res.normalizedQuad[0].x, y: res.normalizedQuad[0].y },
        { x: res.normalizedQuad[1].x, y: res.normalizedQuad[1].y },
        { x: res.normalizedQuad[2].x, y: res.normalizedQuad[2].y },
        { x: res.normalizedQuad[3].x, y: res.normalizedQuad[3].y },
      ]);
      setAutoDetected(res.autoDetected);
      setDetectedMode(res.mode);
    } catch {
      handleResetToInset();
    } finally {
      setIsDetecting(false);
    }
  };

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
      const res = await dewarpImageWithQuad(imageSource, ordered, {
        maxWidth: 1600,
        maxHeight: 1600,
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

  return (
    <div className="space-y-3 select-none">
      {/* Main Interactive Dewarp Canvas Surface */}
      <div
        ref={containerRef}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        className="relative rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-950 aspect-[3/4] sm:aspect-[4/3] max-h-[54vh] min-h-[300px] w-full flex items-center justify-center touch-none"
      >
        <img
          ref={imgRef}
          src={imageSource}
          alt="Document to dewarp"
          onLoad={updateLayoutMetrics}
          className="w-full h-full object-contain pointer-events-none"
        />

        {/* Loading overlay for detection / dewarping */}
        {(isDetecting || isDewarping) && (
          <div className="absolute inset-0 bg-black/65 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2.5 text-zinc-200 text-xs p-4 text-center z-30">
            <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
            <span className="font-medium text-sm">
              {isDewarping ? 'Dewarping & flattening document...' : 'Finding document edges on device...'}
            </span>
            <span className="text-[11px] text-zinc-400">
              {isDewarping ? 'Rectifying perspective distortion' : 'Tracing paper perimeter against background'}
            </span>
          </div>
        )}

        {/* SVG Boundary Layer */}
        {displayedWidth > 0 && displayedHeight > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
            {/* Shaded mask over the exterior */}
            <defs>
              <mask id="quad-crop-mask">
                <rect width="100%" height="100%" fill="white" />
                <polygon points={svgPoints} fill="black" />
              </mask>
            </defs>

            {/* Dark scrim outside the document quad */}
            <rect
              width="100%"
              height="100%"
              fill="rgba(0, 0, 0, 0.45)"
              mask="url(#quad-crop-mask)"
            />

            {/* Cyan boundary lines */}
            <polygon
              points={svgPoints}
              className="fill-cyan-500/10 stroke-cyan-400 stroke-2"
              strokeDasharray="4 2"
            />
          </svg>
        )}

        {/* 4 Interactive Corner Pins */}
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
                className={`absolute -translate-x-1/2 -translate-y-1/2 min-h-[48px] min-w-[48px] flex items-center justify-center cursor-grab active:cursor-grabbing z-20 touch-none group`}
                aria-label={`Drag ${cornerLabels[idx]} corner`}
              >
                {/* Visual pin target */}
                <div
                  className={`w-5 h-5 rounded-full border-2 transition-transform ${
                    isActive
                      ? 'scale-125 bg-white border-cyan-400 ring-4 ring-cyan-400/50 shadow-lg'
                      : 'bg-white border-cyan-500 shadow-md group-hover:scale-110'
                  }`}
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-600 m-auto mt-1" />
                </div>
              </button>
            );
          })}

        {/* Floating 2.5x Magnification Loupe HUD (offset above active thumb) */}
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

        {/* Top Overlay Badge & Action */}
        <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 z-30 pointer-events-auto">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-950/85 text-cyan-300 text-[11px] font-medium backdrop-blur-sm border border-cyan-500/30">
            <Crop className="w-3.5 h-3.5" />
            <span>
              {autoDetected
                ? detectedMode === 'sheet'
                  ? 'Sheet Perimeter Detected'
                  : 'Text Block Auto-Anchored'
                : 'Adjust 4 corners to fit text'}
            </span>
          </span>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={handleReDetect}
              className="min-h-[36px] px-2.5 rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700 text-xs font-medium inline-flex items-center gap-1 whitespace-nowrap"
              title="Auto-detect document edges"
            >
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Auto</span>
            </button>
            {onRetake && (
              <button
                type="button"
                onClick={onRetake}
                className="min-h-[36px] px-2.5 rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700 text-xs font-medium inline-flex items-center gap-1 whitespace-nowrap"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Retake</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Natural Thumb-Zone Controls Bar (Mobile >= 44px hitboxes) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 px-0.5">
          <span>Drag corner pins to frame paper sheet:</span>
          <button
            type="button"
            onClick={handleResetToInset}
            className="text-cyan-600 dark:text-cyan-400 hover:underline"
          >
            Reset Corners
          </button>
        </div>

        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            disabled={isDewarping || isDetecting}
            onClick={handleExecuteDewarp}
            className="min-h-[48px] flex-1 px-4 rounded-xl bg-cyan-600 hover:bg-cyan-700 active:scale-[0.98] disabled:opacity-50 text-white text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all shadow-sm"
          >
            {isDewarping ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Check className="w-4 h-4" />
            )}
            <span>Dewarp & Read Note</span>
          </button>

          <button
            type="button"
            disabled={isDewarping}
            onClick={onCancelOrFull}
            className="min-h-[48px] px-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
          >
            <span>Skip Dewarp</span>
          </button>
        </div>
      </div>
    </div>
  );
};
