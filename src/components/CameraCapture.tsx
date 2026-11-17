import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera,
  RefreshCw,
  X,
  Upload,
  Loader2,
  Image as ImageIcon,
  Crop,
  Check,
  Square,
  Sparkles,
} from 'lucide-react';
import {
  detectWordsOnImage,
  cropImageToSquareDataUrl,
  WordDetectionResult,
  DetectedWordSquare,
} from '../lib/ocr/onDeviceWordDetector';
import { InteractiveQuadCropper } from './InteractiveQuadCropper';

interface CameraCaptureProps {
  onCapture: (dataUrl: string) => void;
  onClose?: () => void;
  embedded?: boolean;
  paused?: boolean;
  onUploadFile?: (file: File) => void;
  statusNote?: string;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({
  onCapture,
  onClose,
  embedded = false,
  paused = false,
  onUploadFile,
  statusNote,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement | null>(null);
  const galleryInputRef = useRef<HTMLInputElement | null>(null);

  const mountedRef = useRef(true);
  const pausedRef = useRef(paused);
  const attemptIdRef = useRef(0);
  const selectedDeviceIdRef = useRef('');
  const facingModeRef = useRef<'environment' | 'user'>('environment');

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isStarting, setIsStarting] = useState(!paused);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // On-device perspective quad dewarp and versatile word-edge detection states
  const [autoDewarp, setAutoDewarp] = useState(true);
  const [pendingDewarpImage, setPendingDewarpImage] = useState<string | null>(null);
  const [autoSquareCrop, setAutoSquareCrop] = useState(true);
  const [pendingCropImage, setPendingCropImage] = useState<string | null>(null);
  const [wordDetection, setWordDetection] = useState<WordDetectionResult | null>(null);
  const [selectedWord, setSelectedWord] = useState<DetectedWordSquare | null>(null);
  const [isDetectingWords, setIsDetectingWords] = useState(false);
  const [isApplyingCrop, setIsApplyingCrop] = useState(false);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceId;
  }, [selectedDeviceId]);

  useEffect(() => {
    facingModeRef.current = facingMode;
  }, [facingMode]);

  const stopTracks = (stream: MediaStream | null) => {
    if (!stream) return;
    stream.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch {
        // ignore track stop failure
      }
    });
  };

  const stopStream = useCallback(() => {
    attemptIdRef.current += 1;
    if (streamRef.current) {
      stopTracks(streamRef.current);
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      if (!mountedRef.current) return;
      const videoInputs = all.filter((d) => d.kind === 'videoinput');
      setDevices(videoInputs);
    } catch {
      // ignore enumeration errors
    }
  }, []);

  const startCamera = useCallback(
    async (opts?: { deviceId?: string; facing?: 'environment' | 'user' }) => {
      stopStream();
      if (!mountedRef.current || pausedRef.current || document.visibilityState === 'hidden') {
        setIsStarting(false);
        return;
      }

      const myAttempt = ++attemptIdRef.current;
      setIsStarting(true);
      setErrorMsg(null);

      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        if (mountedRef.current && attemptIdRef.current === myAttempt) {
          setIsStarting(false);
          setErrorMsg(
            'Live camera streaming is unavailable in this browser context. Choose a photo, drop an image here, or open your device camera.'
          );
        }
        return;
      }

      const targetDeviceId =
        opts?.deviceId !== undefined ? opts.deviceId : selectedDeviceIdRef.current;
      const targetFacing = opts?.facing ?? facingModeRef.current;

      // Progressive constraint fallback chain so any phone, tablet, laptop, or webcam works
      const constraintCandidates: MediaStreamConstraints[] = targetDeviceId
        ? [
            {
              video: {
                deviceId: { exact: targetDeviceId },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
              },
              audio: false,
            },
            {
              video: { deviceId: { exact: targetDeviceId } },
              audio: false,
            },
            { video: true, audio: false },
          ]
        : [
            {
              video: {
                facingMode: { ideal: targetFacing },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
              },
              audio: false,
            },
            {
              video: { facingMode: targetFacing },
              audio: false,
            },
            { video: true, audio: false },
          ];

      let acquiredStream: MediaStream | null = null;
      let lastErr: any = null;

      for (const constraints of constraintCandidates) {
        if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
          break;
        }
        try {
          acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (err: any) {
          lastErr = err;
          if (
            err?.name === 'NotAllowedError' ||
            err?.name === 'PermissionDeniedError' ||
            err?.name === 'NotReadableError'
          ) {
            break;
          }
        }
      }

      // Async unmount / paused / superseded guard so tracks never leak in background
      if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
        stopTracks(acquiredStream);
        if (mountedRef.current) {
          setIsStarting(false);
        }
        return;
      }

      if (!acquiredStream) {
        setIsStarting(false);
        const name = lastErr?.name || '';
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
          setErrorMsg(
            'Camera permission was denied. Allow camera access in browser settings, or pick/drop an image below.'
          );
        } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
          setErrorMsg(
            'No live camera was found on this device. Drop an image here, paste a screenshot, or browse your photos.'
          );
        } else if (name === 'NotReadableError') {
          setErrorMsg(
            'Camera is currently used by another app. Close the other app and tap Retry Camera, or choose a photo below.'
          );
        } else {
          setErrorMsg(
            lastErr?.message ||
              'Could not start live camera preview. Choose a photo, drop an image, or use device capture.'
          );
        }
        return;
      }

      streamRef.current = acquiredStream;

      const activeTrack = acquiredStream.getVideoTracks()[0];
      const activeSettings = activeTrack?.getSettings?.();
      if (activeSettings?.deviceId && !targetDeviceId) {
        setSelectedDeviceId(activeSettings.deviceId);
        selectedDeviceIdRef.current = activeSettings.deviceId;
      }

      await refreshDevices();
      if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
        stopTracks(acquiredStream);
        streamRef.current = null;
        return;
      }

      if (videoRef.current) {
        videoRef.current.srcObject = acquiredStream;
        try {
          await videoRef.current.play();
        } catch {
          // autoPlay + playsInline will render frames on browsers blocking programmatic play
        }
      }

      if (mountedRef.current && attemptIdRef.current === myAttempt) {
        setIsStarting(false);
      }
    },
    [refreshDevices, stopStream]
  );

  // Lifecycle: mount/unmount + pause/resume when review sheet or lightbox opens
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopStream();
    };
  }, [stopStream]);

  useEffect(() => {
    if (paused) {
      stopStream();
      setIsStarting(false);
    } else if (document.visibilityState !== 'hidden') {
      startCamera();
    }
  }, [paused, startCamera, stopStream]);

  // Pause camera hardware when browser tab goes background; resume cleanly on return
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        stopStream();
        setIsStarting(false);
      } else if (!pausedRef.current && mountedRef.current) {
        startCamera();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [startCamera, stopStream]);

  const handleSwitchDevice = (nextDeviceId: string) => {
    setSelectedDeviceId(nextDeviceId);
    selectedDeviceIdRef.current = nextDeviceId;
    startCamera({ deviceId: nextDeviceId });
  };

  const handleFlipCamera = () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    facingModeRef.current = nextFacing;
    setSelectedDeviceId('');
    selectedDeviceIdRef.current = '';
    startCamera({ deviceId: '', facing: nextFacing });
  };

  const handleRunWordDetection = async (dataUrl: string) => {
    setPendingCropImage(dataUrl);
    setIsDetectingWords(true);
    setSelectedWord(null);

    try {
      const result = await detectWordsOnImage(dataUrl, {
        maxDimension: 960,
        generateThumbnails: true,
        maxThumbnails: 24,
      });

      if (result.words.length > 0) {
        setWordDetection(result);
      } else {
        // Fallback directly to full frame if no words found
        setPendingCropImage(null);
        setWordDetection(null);
        onCapture(dataUrl);
      }
    } catch (err) {
      console.warn('On-device word detection skipped:', err);
      setPendingCropImage(null);
      setWordDetection(null);
      onCapture(dataUrl);
    } finally {
      setIsDetectingWords(false);
    }
  };

  const handleCapturedImageReady = (dataUrl: string) => {
    stopStream();
    if (autoDewarp) {
      setPendingDewarpImage(dataUrl);
    } else if (autoSquareCrop) {
      handleRunWordDetection(dataUrl);
    } else {
      onCapture(dataUrl);
    }
  };

  const handleApplyAllWordsSquareCrop = async () => {
    if (!pendingCropImage || !wordDetection?.documentSquareBox) {
      if (pendingCropImage) onCapture(pendingCropImage);
      setPendingCropImage(null);
      setWordDetection(null);
      return;
    }
    setIsApplyingCrop(true);
    try {
      const box = {
        x0: wordDetection.documentSquareBox.x0,
        y0: wordDetection.documentSquareBox.y0,
        x1: wordDetection.documentSquareBox.x1,
        y1: wordDetection.documentSquareBox.y1,
      };
      const squareDataUrl = await cropImageToSquareDataUrl(pendingCropImage, box, 768, 0.12);
      setPendingCropImage(null);
      setWordDetection(null);
      onCapture(squareDataUrl);
    } catch {
      onCapture(pendingCropImage);
      setPendingCropImage(null);
      setWordDetection(null);
    } finally {
      setIsApplyingCrop(false);
    }
  };

  const handleApplySingleWordSquareCrop = async (word: DetectedWordSquare) => {
    if (!pendingCropImage) return;
    setIsApplyingCrop(true);
    try {
      const squareDataUrl = await cropImageToSquareDataUrl(pendingCropImage, word.box, 480, 0.22);
      setPendingCropImage(null);
      setWordDetection(null);
      onCapture(squareDataUrl);
    } catch {
      onCapture(pendingCropImage);
      setPendingCropImage(null);
      setWordDetection(null);
    } finally {
      setIsApplyingCrop(false);
    }
  };

  const handleKeepFullPhoto = () => {
    if (pendingCropImage) {
      onCapture(pendingCropImage);
    }
    setPendingCropImage(null);
    setWordDetection(null);
  };

  const handleRetake = () => {
    setPendingDewarpImage(null);
    setPendingCropImage(null);
    setWordDetection(null);
    setSelectedWord(null);
    startCamera();
  };

  const handleCaptureFrame = () => {
    const video = videoRef.current;
    if (
      isCapturing ||
      !video ||
      video.readyState < 2 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    setIsCapturing(true);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Always draw unmirrored raw frame so text is never backwards for OCR
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
      handleCapturedImageReady(dataUrl);
    } finally {
      setIsCapturing(false);
    }
  };

  const processSelectedFile = (file: File) => {
    if (onUploadFile) {
      stopStream();
      onUploadFile(file);
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const result = ev.target?.result as string;
      if (result) {
        handleCapturedImageReady(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
    e.target.value = '';
  };

  const liveReady = !isStarting && !errorMsg && !paused;

  // When a photo has been captured or selected and autoDewarp is active, present the interactive 4-corner quad cropper
  if (pendingDewarpImage) {
    return (
      <div className="space-y-3">
        <InteractiveQuadCropper
          imageSource={pendingDewarpImage}
          onApplyDewarp={(dewarpedDataUrl) => {
            setPendingDewarpImage(null);
            if (autoSquareCrop) {
              handleRunWordDetection(dewarpedDataUrl);
            } else {
              onCapture(dewarpedDataUrl);
            }
          }}
          onCancelOrFull={() => {
            const raw = pendingDewarpImage;
            setPendingDewarpImage(null);
            if (autoSquareCrop) {
              handleRunWordDetection(raw);
            } else {
              onCapture(raw);
            }
          }}
          onRetake={handleRetake}
        />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Main Viewfinder Surface (supports live camera + on-device word crop + drag-and-drop) */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file && file.type.startsWith('image/')) {
            processSelectedFile(file);
          }
        }}
        className={`relative rounded-2xl overflow-hidden border transition-colors bg-zinc-950 aspect-[3/4] sm:aspect-[4/3] max-h-[54vh] min-h-[280px] w-full flex items-center justify-center select-none ${
          isDragOver
            ? 'border-indigo-500 ring-2 ring-indigo-500/40'
            : 'border-zinc-200 dark:border-zinc-800'
        }`}
      >
        {pendingCropImage ? (
          <div className="relative w-full h-full flex items-center justify-center bg-zinc-950 overflow-hidden">
            <img
              src={pendingCropImage}
              alt="Note to crop"
              className="w-full h-full object-contain pointer-events-none"
            />

            {/* Detecting words loading overlay */}
            {isDetectingWords && (
              <div className="absolute inset-0 bg-black/65 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2.5 text-zinc-200 text-xs p-4 text-center z-20">
                <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                <span className="font-medium text-sm">Detecting word edges on device...</span>
                <span className="text-[11px] text-zinc-400">
                  Fitting each word into an unskewed square
                </span>
              </div>
            )}

            {/* Interactive detected word boxes and square outlines */}
            {!isDetectingWords && wordDetection && (
              <div className="absolute inset-0 z-10 pointer-events-auto">
                {wordDetection.words.map((word) => {
                  const isSelected = selectedWord?.id === word.id;
                  return (
                    <React.Fragment key={word.id}>
                      {/* Word edge box */}
                      <button
                        type="button"
                        onClick={() => setSelectedWord(isSelected ? null : word)}
                        style={{
                          left: `${word.normalized.word.x * 100}%`,
                          top: `${word.normalized.word.y * 100}%`,
                          width: `${word.normalized.word.width * 100}%`,
                          height: `${word.normalized.word.height * 100}%`,
                        }}
                        className={`absolute border transition-all rounded-[2px] cursor-pointer ${
                          isSelected
                            ? 'border-cyan-400 bg-cyan-400/35 ring-2 ring-cyan-400/80 z-20'
                            : 'border-indigo-400/85 bg-indigo-500/20 hover:border-cyan-300 hover:bg-cyan-500/30 z-10'
                        }`}
                        title={`Word ${word.wordIndex + 1} (Tap to fit square)`}
                      />

                      {/* Selected word square boundary */}
                      {isSelected && (
                        <div
                          style={{
                            left: `${word.normalized.square.x * 100}%`,
                            top: `${word.normalized.square.y * 100}%`,
                            width: `${word.normalized.square.width * 100}%`,
                            height: `${word.normalized.square.height * 100}%`,
                          }}
                          className="absolute border-2 border-dashed border-cyan-400 bg-cyan-500/10 rounded-lg pointer-events-none z-15 animate-pulse"
                        />
                      )}
                    </React.Fragment>
                  );
                })}

                {/* Overall document square enclosing all words */}
                {!selectedWord && wordDetection.documentSquareBox && (
                  <div
                    style={{
                      left: `${wordDetection.documentSquareBox.normalized.x * 100}%`,
                      top: `${wordDetection.documentSquareBox.normalized.y * 100}%`,
                      width: `${wordDetection.documentSquareBox.normalized.width * 100}%`,
                      height: `${wordDetection.documentSquareBox.normalized.height * 100}%`,
                    }}
                    className="absolute border-2 border-dashed border-indigo-400/85 bg-indigo-500/10 rounded-xl pointer-events-none z-15"
                  />
                )}
              </div>
            )}
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover transition-opacity duration-200 ${
              facingMode === 'user' ? 'scale-x-[-1]' : ''
            } ${liveReady ? 'opacity-100' : 'opacity-0'}`}
          />
        )}

        {/* Subtle alignment reticle guides when camera stream is active */}
        {!pendingCropImage && liveReady && (
          <div className="pointer-events-none absolute inset-6 sm:inset-8">
            <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-white/55 rounded-tl-md" />
            <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-white/55 rounded-tr-md" />
            <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-white/55 rounded-bl-md" />
            <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-white/55 rounded-br-md" />
          </div>
        )}

        {/* Drag-and-drop overlay indicator */}
        {isDragOver && (
          <div className="absolute inset-0 z-20 bg-indigo-950/80 flex flex-col items-center justify-center gap-2 text-white p-4 text-center">
            <Upload className="w-7 h-7 text-indigo-300" />
            <p className="text-sm font-medium">Drop image to auto-scan offline</p>
          </div>
        )}

        {/* Starting / Paused state */}
        {!pendingCropImage && isStarting && !errorMsg && !paused && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2.5 text-zinc-300 text-xs p-4 text-center">
            <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
            <span>Starting live camera...</span>
          </div>
        )}

        {!pendingCropImage && paused && !errorMsg && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-zinc-400 text-xs p-4 text-center bg-zinc-950">
            <Camera className="w-5 h-5 opacity-60" />
            <span>Camera paused while reviewing scan</span>
          </div>
        )}

        {/* Camera unavailable / permission denied interactive fallback */}
        {!pendingCropImage && errorMsg && (
          <div className="absolute inset-0 p-5 flex flex-col items-center justify-center text-center gap-3.5 bg-zinc-950/95">
            <div className="w-11 h-11 rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-300 flex items-center justify-center">
              <Camera className="w-5 h-5" />
            </div>
            <p className="text-xs sm:text-sm text-zinc-300 max-w-sm leading-relaxed">{errorMsg}</p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={() => startCamera()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-100 text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Camera</span>
              </button>
              <button
                type="button"
                onClick={() => galleryInputRef.current?.click()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Choose Photo</span>
              </button>
              <button
                type="button"
                onClick={() => nativeInputRef.current?.click()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Device Cam</span>
              </button>
            </div>
            <p className="text-[11px] text-zinc-500">
              Or drop an image file here or paste a screenshot (Ctrl+V)
            </p>
          </div>
        )}

        {/* Top Bar Overlay */}
        <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 z-30">
          {pendingCropImage ? (
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-950/85 text-cyan-300 text-[11px] font-medium backdrop-blur-sm border border-cyan-500/30">
                <Crop className="w-3.5 h-3.5" />
                <span>
                  {wordDetection
                    ? `${wordDetection.words.length} words detected · Square Fitted`
                    : 'Detecting word edges...'}
                </span>
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 min-w-0">
              {devices.length > 1 ? (
                <select
                  value={selectedDeviceId}
                  onChange={(e) => handleSwitchDevice(e.target.value)}
                  aria-label="Select camera device"
                  className="min-h-[40px] max-w-[190px] px-2.5 rounded-xl bg-zinc-900/85 text-zinc-100 border border-zinc-700 text-xs font-medium focus:outline-none truncate"
                >
                  {devices.map((dev, idx) => (
                    <option key={dev.deviceId || idx} value={dev.deviceId}>
                      {dev.label || `Camera ${idx + 1}`}
                    </option>
                  ))}
                </select>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-950/75 text-zinc-200 text-[11px] font-medium backdrop-blur-sm">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      liveReady ? 'bg-emerald-400' : 'bg-zinc-500'
                    }`}
                  />
                  <span>{statusNote || 'Auto-detects printed & handwriting'}</span>
                </span>
              )}
            </div>
          )}

          {pendingCropImage ? (
            <button
              type="button"
              onClick={handleRetake}
              className="min-h-[36px] px-3 rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700 text-xs font-medium inline-flex items-center gap-1 whitespace-nowrap shrink-0"
              aria-label="Retake photo"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retake</span>
            </button>
          ) : (
            !embedded &&
            onClose && (
              <button
                type="button"
                onClick={() => {
                  stopStream();
                  onClose();
                }}
                className="min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700 shrink-0"
                aria-label="Close camera"
              >
                <X className="w-4 h-4" />
              </button>
            )
          )}
        </div>
      </div>

      {/* Hidden inputs: Gallery upload (all image files) & Native Device Camera fallback */}
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        className="hidden"
      />
      <input
        ref={nativeInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Action Controls Area */}
      {pendingCropImage ? (
        <div className="space-y-2.5">
          {/* Scrollable Word Squares Carousel */}
          {wordDetection && wordDetection.words.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 px-0.5">
                <span>Tap a word to isolate its square crop:</span>
                <span className="font-mono-nums">
                  {selectedWord ? `Word ${selectedWord.wordIndex + 1} focused` : 'All notes square'}
                </span>
              </div>
              <div className="flex items-center gap-2 overflow-x-auto pb-1 pt-0.5 no-scrollbar">
                <button
                  type="button"
                  onClick={() => setSelectedWord(null)}
                  className={`min-h-[44px] px-3 rounded-xl border text-xs font-medium inline-flex items-center gap-1.5 shrink-0 transition-colors ${
                    !selectedWord
                      ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300'
                      : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400'
                  }`}
                >
                  <Square className="w-3.5 h-3.5" />
                  <span>All Notes Square</span>
                </button>

                {wordDetection.words.map((word, idx) => {
                  const isSelected = selectedWord?.id === word.id;
                  return (
                    <button
                      key={word.id}
                      type="button"
                      onClick={() => setSelectedWord(isSelected ? null : word)}
                      className={`min-h-[44px] px-2 py-1 rounded-xl border flex items-center gap-2 shrink-0 transition-colors ${
                        isSelected
                          ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-950/60 text-cyan-700 dark:text-cyan-300 ring-2 ring-cyan-500/30'
                          : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300'
                      }`}
                    >
                      {word.thumbnailUrl ? (
                        <img
                          src={word.thumbnailUrl}
                          alt={`Word ${idx + 1}`}
                          className="w-7 h-7 rounded object-cover border border-zinc-200 dark:border-zinc-700 shrink-0"
                        />
                      ) : (
                        <Square className="w-3.5 h-3.5 text-zinc-400" />
                      )}
                      <span className="text-[11px] font-mono-nums whitespace-nowrap">
                        W{word.wordIndex + 1}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Primary Action Buttons */}
          <div className="flex items-center justify-between gap-2">
            {selectedWord ? (
              <button
                type="button"
                disabled={isApplyingCrop}
                onClick={() => handleApplySingleWordSquareCrop(selectedWord)}
                className="min-h-[48px] flex-1 px-4 rounded-xl bg-cyan-600 hover:bg-cyan-700 active:scale-[0.98] disabled:opacity-50 text-white text-xs sm:text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all shadow-sm"
              >
                {isApplyingCrop ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Crop className="w-4 h-4" />
                )}
                <span>Square Crop Word {selectedWord.wordIndex + 1}</span>
              </button>
            ) : (
              <button
                type="button"
                disabled={isApplyingCrop}
                onClick={handleApplyAllWordsSquareCrop}
                className="min-h-[48px] flex-1 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] disabled:opacity-50 text-white text-xs sm:text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all shadow-sm"
              >
                {isApplyingCrop ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Crop className="w-4 h-4" />
                )}
                <span>Square Crop All Words</span>
              </button>
            )}

            <button
              type="button"
              disabled={isApplyingCrop}
              onClick={() => {
                if (pendingCropImage) {
                  setPendingDewarpImage(pendingCropImage);
                  setPendingCropImage(null);
                  setWordDetection(null);
                  setSelectedWord(null);
                }
              }}
              className="min-h-[48px] px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-cyan-600 dark:text-cyan-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
              title="Adjust 4 document corners"
            >
              <Crop className="w-3.5 h-3.5" />
              <span>Dewarp</span>
            </button>

            <button
              type="button"
              disabled={isApplyingCrop}
              onClick={handleKeepFullPhoto}
              className="min-h-[48px] px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <span>Full Photo</span>
            </button>
          </div>
        </div>
      ) : (
        /* Standard Viewfinder Action Bar */
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={handleFlipCamera}
              disabled={paused}
              className="min-h-[44px] px-3 sm:px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
              aria-label="Flip camera"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Flip</span>
            </button>

            <button
              type="button"
              onClick={handleCaptureFrame}
              disabled={!liveReady || isCapturing}
              className="min-h-[48px] flex-1 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] disabled:opacity-50 text-white text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all"
            >
              <Camera className="w-4 h-4 shrink-0" />
              <span>Take Photo</span>
            </button>

            <button
              type="button"
              onClick={() => galleryInputRef.current?.click()}
              className="min-h-[44px] px-3 sm:px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
              aria-label="Choose photo from gallery"
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>Gallery</span>
            </button>
          </div>

          {/* Edge Dewarp & Square-Crop Mode Toggles + Native OS Cam */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                type="button"
                onClick={() => setAutoDewarp((v) => !v)}
                className={`min-h-[32px] px-2.5 py-1 rounded-lg border text-[11px] font-medium inline-flex items-center gap-1.5 transition-colors ${
                  autoDewarp
                    ? 'border-cyan-500/50 bg-cyan-50 dark:bg-cyan-950/40 text-cyan-700 dark:text-cyan-300'
                    : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400'
                }`}
                title="Automatically detect 4 document edges and rectify perspective distortion"
              >
                <Sparkles className="w-3 h-3 text-cyan-500" />
                <span>{autoDewarp ? 'Auto-Dewarp: ON' : 'Dewarp: OFF'}</span>
              </button>

              <button
                type="button"
                onClick={() => setAutoSquareCrop((v) => !v)}
                className={`min-h-[32px] px-2.5 py-1 rounded-lg border text-[11px] font-medium inline-flex items-center gap-1.5 transition-colors ${
                  autoSquareCrop
                    ? 'border-indigo-500/50 bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300'
                    : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400'
                }`}
                title="Detect word bounding edges and fit into square crops"
              >
                <Crop className="w-3 h-3" />
                <span>{autoSquareCrop ? 'Words Square: ON' : 'Words: OFF'}</span>
              </button>
            </div>

            <button
              type="button"
              onClick={() => nativeInputRef.current?.click()}
              className="min-h-[32px] inline-flex items-center gap-1 hover:text-zinc-900 dark:hover:text-zinc-200 underline underline-offset-2"
            >
              <span>Native camera</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
