import mongoose from 'mongoose';

const resetRequestSchema = new mongoose.Schema({
  email: { type: String, required: true },
  desiredPassword: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});


const ResetRequest = mongoose.model('ResetRequest', resetRequestSchema);

export default ResetRequest;
