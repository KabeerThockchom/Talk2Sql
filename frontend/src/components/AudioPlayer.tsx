import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';

interface AudioPlayerProps {
  audioBase64?: string;
  className?: string;
}

export function AudioPlayer({ audioBase64, className = '' }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  useEffect(() => {
    // Reset state when audio changes
    setIsPlaying(false);
    setCurrentTime(0);
    
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [audioBase64]);
  
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const setAudioData = () => {
      setDuration(audio.duration);
    };
    
    const setAudioTime = () => {
      setCurrentTime(audio.currentTime);
    };
    
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };
    
    // Add event listeners
    audio.addEventListener('loadeddata', setAudioData);
    audio.addEventListener('timeupdate', setAudioTime);
    audio.addEventListener('ended', handleEnded);
    
    // Clean up
    return () => {
      audio.removeEventListener('loadeddata', setAudioData);
      audio.removeEventListener('timeupdate', setAudioTime);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioRef]);
  
  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;
    
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    
    setIsPlaying(!isPlaying);
  };
  
  const formatTime = (time: number) => {
    if (isNaN(time)) return '0:00';
    
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };
  
  const handleProgress = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };
  
  if (!audioBase64) {
    return null;
  }
  
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <audio
        ref={audioRef}
        src={`data:audio/wav;base64,${audioBase64}`}
        preload="metadata"
      />
      
      <button
        onClick={togglePlayPause}
        className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-white hover:bg-primary-hover transition-colors"
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <Pause className="w-5 h-5" />
        ) : (
          <Play className="w-5 h-5" />
        )}
      </button>
      
      <div className="flex-1 flex items-center gap-2">
        <span className="text-xs text-neutral-600 dark:text-neutral-400 min-w-[40px]">
          {formatTime(currentTime)}
        </span>
        
        <input
          type="range"
          min="0"
          max={duration || 0}
          value={currentTime}
          onChange={handleProgress}
          className="flex-1 h-1 bg-neutral-200 dark:bg-neutral-700 rounded-full cursor-pointer accent-primary"
        />
        
        <span className="text-xs text-neutral-600 dark:text-neutral-400 min-w-[40px]">
          {formatTime(duration)}
        </span>
      </div>
      
      <Volume2 className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
    </div>
  );
} 