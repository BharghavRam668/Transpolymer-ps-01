// adminSchema.js
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const adminSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true }, // changed from username
  passwordHash: { type: String, required: true },
});

// Method to check password
adminSchema.methods.comparePassword = async function (password) {
  return await bcrypt.compare(password, this.passwordHash);
};

export default mongoose.model('Admin', adminSchema);
