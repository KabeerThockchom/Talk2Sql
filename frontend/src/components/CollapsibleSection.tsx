import React, { useState, ReactNode } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Tooltip } from './Tooltip';

interface CollapsibleSectionProps {
  title: string;
  children: ReactNode;
  defaultExpanded?: boolean;
  icon?: ReactNode;
  className?: string;
}

export function CollapsibleSection({
  title,
  children,
  defaultExpanded = true,
  icon,
  className = '',
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const toggleExpanded = () => setIsExpanded(prev => !prev);

  return (
    <div className={`bg-white dark:bg-neutral-800 rounded-lg shadow-card overflow-hidden ${className}`}>
      <Tooltip content={isExpanded ? `Collapse ${title}` : `Expand ${title}`}>
        <button
          onClick={toggleExpanded}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-neutral-300 hover:text-black dark:hover:bg-neutral-300 dark:hover:text-black transition-colors"
        >
          <div className="flex items-center gap-2">
            {icon}
            <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">{title}</h3>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
          )}
        </button>
      </Tooltip>
      
      {isExpanded && (
        <div className="p-4 border-t border-neutral-200 dark:border-neutral-700">
          {children}
        </div>
      )}
    </div>
  );
} 