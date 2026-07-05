export async function analyzeAudio(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    // Try to extract the detailed error message returned by FastAPI
    try {
      const errorBody = await res.json();
      const detail = errorBody?.detail ?? `Server error ${res.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    } catch (jsonErr) {
      // If the body isn't valid JSON, fall back to status text
      throw new Error(`Analysis failed (HTTP ${res.status}: ${res.statusText})`);
    }
  }
  return res.json();
}
