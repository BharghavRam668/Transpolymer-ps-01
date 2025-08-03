import express from 'express';
import userMiddleware from '../middleware/userMiddleware.js';
import SearchHistory from '../model/searchHistorySchema.js';

const router = express.Router();

/* save a search */
router.post('/', userMiddleware, async (req, res) => {
  const { smiles } = req.body;
  if (!smiles) return res.status(400).json({ message: 'SMILES is required' });
  try {
    await SearchHistory.create({ user_id: req.user.id, smiles });
    res.status(201).json({ message: 'saved' });
  } catch (e) {
    console.error(e); res.status(500).json({ message: 'DB error' });
  }
});

/* get recent searches */
router.get('/', userMiddleware, async (req, res) => {
  try {
    const list = await SearchHistory.find({ user_id: req.user.id })
                    .sort({ created_at: -1 }).limit(10);
    res.json(list);
  } catch (e) {
    console.error(e); res.status(500).json({ message: 'DB error' });
  }
});

export default router;
