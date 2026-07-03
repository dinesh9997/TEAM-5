import { useState, useRef, useEffect } from "react";
import { Mic, Square, Moon, Sun, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Progress } from "@/components/ui/progress";
import { analyzeAudio } from "./api/backend";
import type { AnalysisResult } from "./types/analysis";

function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(() => {
    const stored = localStorage.getItem('darkMode');
    if (stored !== null) return stored === 'true';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const durationIntervalRef = useRef<number | null>(null);

  // Apply dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', String(darkMode));
  }, [darkMode]);

  async function startRecording() {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.start();
      setRecording(true);
      setDuration(0);

      // Clear any previous interval to prevent double-counting
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
      durationIntervalRef.current = window.setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);
    } catch (err) {
      setError("Failed to access microphone. Please grant permission and try again.");
      console.error("Recording error:", err);
    }
  }

  async function stopRecording() {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;

    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }

    recorder.stop();
    recorder.stream.getTracks().forEach(track => track.stop());

    recorder.onstop = async () => {
      const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
      const file = new File([audioBlob], "recording.wav", { type: "audio/wav" });

      setLoading(true);
      setRecording(false);

      try {
        const analysisResult = await analyzeAudio(file);
        setResult(analysisResult);
        setError(null);
      } catch (err) {
        setError("Analysis failed. Please try again.");
        console.error("Analysis error:", err);
      } finally {
        setLoading(false);
      }
    };
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const normalizeScore = (score: number) => {
    return score <= 1 ? score * 100 : score;
  };

  const formatConfidenceScore = (score: number) => {
    return normalizeScore(score).toFixed(0) + '%';
  };

  const getConfidenceColor = (score: number) => {
    const normalized = normalizeScore(score);
    if (normalized >= 80) return "bg-green-500";
    if (normalized >= 60) return "bg-amber-500";
    return "bg-red-500";
  };

  const getConfidenceVariant = (score: number): 'success' | 'warning' | 'destructive' => {
    const normalized = normalizeScore(score);
    if (normalized >= 80) return "success";
    if (normalized >= 60) return "warning";
    return "destructive";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 transition-colors">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-teal-500 to-teal-700 rounded-lg flex items-center justify-center text-white text-xl">
                🎙️
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  AI Speech Analysis
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Multi-Agent Personality Insights
                </p>
              </div>
            </div>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setDarkMode(!darkMode)}
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <Sun className="h-5 w-5 text-yellow-500" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        {/* Info Alert */}
        <Card className="mb-6 border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2 text-blue-900 dark:text-blue-300">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              How It Works
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              Record your voice for 30-60 seconds. Our AI system uses <strong>Faster-Whisper</strong> for transcription,
              analyzes acoustic features, and employs <strong>multi-agent AI</strong> for personality insights enhanced
              with <strong>RAG</strong> knowledge retrieval.
            </p>
          </CardContent>
        </Card>

        {/* Recording Card */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Mic className="w-5 h-5" />
                Voice Recorder
              </CardTitle>
              {recording && (
                <Badge variant="destructive">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></span>
                    Recording {formatDuration(duration)}
                  </span>
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-center">
              {!recording ? (
                <Button
                  size="lg"
                  onClick={startRecording}
                  disabled={loading}
                  className="min-w-[200px]"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Mic className="mr-2 h-5 w-5" />
                      Start Recording
                    </>
                  )}
                </Button>
              ) : (
                <Button
                  size="lg"
                  variant="destructive"
                  onClick={stopRecording}
                  className="min-w-[200px]"
                >
                  <Square className="mr-2 h-5 w-5" />
                  Stop Recording
                </Button>
              )}
            </div>

            {loading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                  <span>Processing audio...</span>
                  <span>Please wait</span>
                </div>
                <Progress value={66} className="h-2" />
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-800 dark:text-red-400">
                <AlertCircle className="h-5 w-5 flex-shrink-0" />
                <span className="text-sm">{error}</span>
              </div>
            )}

            {!recording && !loading && !result && (
              <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  💡 <strong>Tip:</strong> Speak naturally and clearly. The analysis works best with 30-60 seconds of audio.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        {result && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                Your Analysis Results
              </h2>
              <Button variant="outline" size="sm" onClick={() => setResult(null)}>
                New Analysis
              </Button>
            </div>

            <Card>
              <CardHeader className="bg-gradient-to-r from-teal-500 to-teal-700 text-white">
                <div className="flex items-center justify-between">
                  <CardTitle>Analysis Complete</CardTitle>
                  <Badge variant="success" className="bg-white/20 text-white border-white/30">
                    ✓ Done
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-6 space-y-6">
                {/* Confidence Score */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                    Confidence Level
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant={getConfidenceVariant(result.confidence_score)}>
                        {result.confidence_label}
                      </Badge>
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {formatConfidenceScore(result.confidence_score)}
                      </span>
                    </div>
                    <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`absolute inset-y-0 left-0 ${getConfidenceColor(result.confidence_score)} transition-all duration-500`}
                        style={{ width: `${normalizeScore(result.confidence_score)}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Accordion Sections */}
                <Accordion type="single" collapsible defaultValue="transcript">
                  <AccordionItem value="transcript">
                    <AccordionTrigger>
                      <div className="flex items-center gap-2">
                        <span>📝</span>
                        <span className="font-semibold">Transcript</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 text-gray-700 dark:text-gray-300">
                        {result.transcript}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  {result.speech_metrics && Object.keys(result.speech_metrics).length > 0 && (
                    <AccordionItem value="metrics">
                      <AccordionTrigger>
                        <div className="flex items-center gap-2">
                          <span>📊</span>
                          <span className="font-semibold">Speech Metrics</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          {Object.entries(result.speech_metrics).map(([key, value]) => (
                            <div key={key} className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3">
                              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </div>
                              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                                {typeof value === 'number' ? value.toFixed(2) : value}
                              </div>
                            </div>
                          ))}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  )}

                  {result.agent_results && Object.keys(result.agent_results).length > 0 && (
                    <AccordionItem value="agents">
                      <AccordionTrigger>
                        <div className="flex items-center gap-2">
                          <span>🤖</span>
                          <span className="font-semibold">Agent Analysis</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="space-y-3">
                          {Object.entries(result.agent_results).map(([agentName, agentResult]) => (
                            <div key={agentName} className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
                              <h4 className="font-semibold text-blue-900 dark:text-blue-300 text-sm mb-1">
                                {agentName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </h4>
                              <p className="text-sm text-gray-700 dark:text-gray-300">
                                {typeof agentResult === 'string' ? agentResult : JSON.stringify(agentResult, null, 2)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  )}

                  <AccordionItem value="report">
                    <AccordionTrigger>
                      <div className="flex items-center gap-2">
                        <span>📄</span>
                        <span className="font-semibold">Personality Report</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
                        <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap leading-relaxed">
                          {result.final_report}
                        </p>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

                {/* Action Buttons */}
                <div className="flex gap-2 pt-2">
                  <Button variant="outline" className="flex-1" size="sm">
                    📥 Download
                  </Button>
                  <Button variant="outline" className="flex-1" size="sm">
                    🔗 Share
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Features Grid (shown when no results) */}
        {!result && (
          <div className="mt-8">
            <h3 className="text-center text-lg font-bold text-gray-900 dark:text-gray-100 mb-6">
              Key Features
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { icon: "🎤", title: "Real-time Recording", desc: "High-quality audio capture" },
                { icon: "📊", title: "Acoustic Analysis", desc: "Pitch, energy, speech patterns" },
                { icon: "🤖", title: "Multi-Agent AI", desc: "Specialized personality agents" },
                { icon: "🔍", title: "RAG-Enhanced", desc: "Knowledge-augmented insights" },
              ].map((feature, idx) => (
                <Card key={idx}>
                  <CardHeader className="pb-3">
                    <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center text-2xl mb-2">
                      {feature.icon}
                    </div>
                    <CardTitle className="text-sm">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-xs">{feature.desc}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4">
          <p className="text-center text-xs text-gray-600 dark:text-gray-400">
            © 2025 TEAM-5 Speech Analysis • Powered by Multi-Agent AI & RAG
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
