import React, { useState, ReactNode } from 'react';

interface TooltipProps {
  children: ReactNode;
  content: string;
  delay?: number;
  position?: "top" | "bottom" | "left" | "right";
  offset?: number;
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  delay = 200,
  position = "top",
  offset = 0
}) => {
  const [show, setShow] = useState(false);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    const id = setTimeout(() => setShow(true), delay);
    setTimeoutId(id);
  };

  const handleMouseLeave = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
    setShow(false);
  };

  const getPositionStyles = () => {
    let className = "";
    let style: React.CSSProperties = {};

    switch (position) {
      case "top":
        className = "bottom-full";
        style = { marginBottom: `${8 + offset}px` };
        break;
      case "bottom":
        className = "top-full";
        style = { marginTop: `${8 + offset}px` };
        break;
      case "left":
        className = "right-full";
        style = { marginRight: `${8 + offset}px` };
        break;
      case "right":
        className = "left-full";
        style = { marginLeft: `${8 + offset}px` };
        break;
      default:
        className = "bottom-full";
        style = { marginBottom: `${8 + offset}px` };
    }

    return { className, style };
  };

  const getArrowClass = () => {
    switch (position) {
      case "top":
        return "border-t-neutral-800 dark:border-t-neutral-700 border-l-transparent border-r-transparent border-b-transparent bottom-[-6px]";
      case "bottom":
        return "border-b-neutral-800 dark:border-b-neutral-700 border-l-transparent border-r-transparent border-t-transparent top-[-6px]";
      case "left":
        return "border-l-neutral-800 dark:border-l-neutral-700 border-t-transparent border-b-transparent border-r-transparent right-[-6px]";
      case "right":
        return "border-r-neutral-800 dark:border-r-neutral-700 border-t-transparent border-b-transparent border-l-transparent left-[-6px]";
      default:
        return "border-t-neutral-800 dark:border-t-neutral-700 border-l-transparent border-r-transparent border-b-transparent bottom-[-6px]";
    }
  };

  const { className, style } = getPositionStyles();
  const arrowClass = getArrowClass();

  return (
    <div 
      className="relative inline-flex"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      
      {show && (
        <div className="absolute z-50 w-max min-w-full">
          <div className={`absolute ${className} left-1/2 -translate-x-1/2`} style={style}>
            <div className="relative flex flex-col items-center">
              <div className="bg-neutral-800 dark:bg-neutral-700 text-white px-2 py-1 rounded-md text-xs whitespace-nowrap">
                {content}
              </div>
              <div className={`absolute w-0 h-0 border-[6px] ${arrowClass}`}></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 