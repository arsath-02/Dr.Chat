const express = require('express');
const router = express.Router();
const SummarizedHistory = require('../models/SummarizedHistory');

router.post("/saveSummary", async (req, res) => {
    const { userId, sessionId, summarizedHistory, botResponse } = req.body;
  
    if (!userId || !sessionId || !summarizedHistory || !botResponse) {
      return res.status(400).json({ error: "Missing required fields" });
    }
  
    try {
      const summaryDoc = new SummarizedHistory({
        userId,
        sessionId,
        summary: summarizedHistory,
        botResponse,
        timestamp: new Date()
      });
  
      await summaryDoc.save();
      res.json({ message: "Summary saved successfully" });
    } catch (err) {
      console.error("Error saving summarized history:", err);
      res.status(500).json({ error: "Failed to save summarized history" });
    }
  });
  
module.exports = router;
