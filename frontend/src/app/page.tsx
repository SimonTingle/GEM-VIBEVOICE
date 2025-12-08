"use client";
import { useState, useRef, useEffect } from 'react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Mic, Send, Volume2 } from "lucide-react";

export default function Home() {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [status, setStatus] = useState('Ready');
  
  // Audio Queue System
  const audioQueue = useRef<string[]>([]);
  const isPlaying = useRef(false);

  // Auto-scroll to bottom
  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const processQueue = async () => {
    if (isPlaying.current || audioQueue.current.length === 0) return;
    isPlaying.current = true;
    
    const nextUrl = audioQueue.current.shift();
    if (nextUrl) {
      const audio = new Audio(nextUrl);
      audio.onended = () => {
        isPlaying.current = false;
        processQueue(); 
      };
      await audio.play().catch(e => {
        console.error("Playback error", e);
        isPlaying.current = false;
        processQueue();
      });
    } else {
      isPlaying.current = false;
    }
  };

  const handleSend = async () => {
    if (!input) return;
    const userText = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setIsLoading(true);
    setStatus('Thinking...');

    try {
      // 1. Gemini Generation
      const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY;
      if (!apiKey) throw new Error("Gemini API Key missing");
      
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
      
      const result = await model.generateContent(userText);
      const answer = result.response.text();
      
      setMessages(prev => [...prev, { role: 'bot', content: answer }]);
      setStatus('Speaking...');

      // 2. TTS Request (Split by sentences for streaming feel)
      const sentences = answer.match(/[^.!?]+[.!?]+/g) || [answer];
      const ttsBase = process.env.NEXT_PUBLIC_TTS_URL || "http://localhost:8080";
      
      for (const sentence of sentences) {
         if (sentence.trim().length < 2) continue;
         
         try {
           const res = await fetch(`${ttsBase}/speak`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: sentence })
           });
           
           if (res.ok) {
             const blob = await res.blob();
             const url = URL.createObjectURL(blob);
             audioQueue.current.push(url);
             processQueue();
           }
         } catch (err) {
           console.error("TTS Error", err);
         }
      }
      setStatus('Ready');
    } catch (e) {
      console.error(e);
      setStatus('Error');
      setMessages(prev => [...prev, { role: 'system', content: "Error: Check API Keys or Backend connection." }]);
    }
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b p-4 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
          <h1 className="font-bold text-lg">Gemini + VibeVoice</h1>
        </div>
        <div className="text-xs font-mono text-gray-400 uppercase tracking-widest">{status}</div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <Volume2 className="w-12 h-12 mx-auto mb-2 opacity-20" />
            <p>Say hello to start the conversation.</p>
          </div>
        )}
        
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl p-4 shadow-sm ${
              m.role === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none' 
                : m.role === 'system'
                ? 'bg-red-100 text-red-800'
                : 'bg-white border border-gray-100 rounded-bl-none'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {isLoading && status === 'Thinking...' && (
           <div className="flex justify-start">
             <div className="bg-gray-200 text-gray-500 text-sm px-4 py-2 rounded-full animate-bounce">
               ...
             </div>
           </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t">
        <div className="max-w-3xl mx-auto flex gap-3">
          <input 
            className="flex-1 border border-gray-200 bg-gray-50 rounded-full px-6 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
            value={input}
            placeholder="Type your message..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isLoading}
          />
          <button 
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
