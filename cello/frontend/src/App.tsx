import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ViewerPage from "./components/ViewerPage";
import DemoPage from "./components/DemoPage";

/** Deep link: /?image=<imageId>[&triage=1] preselects a slide (and auto-runs triage). */
const initialImage = new URLSearchParams(window.location.search).get("image") ?? undefined;

const App: React.FC = () => {
  const [selectedImageId, setSelectedImageId] = useState<string | undefined>(initialImage);

  // Login/Register pages still exist under components/ but are not routed yet
  // (the backend auth is an in-memory mock). Everything lives on "/".
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ViewerPage selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId} />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="*" element={<ViewerPage selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId} />} />
      </Routes>
    </Router>
  );
};

export default App;
