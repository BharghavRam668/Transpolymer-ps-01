// scripts/createAdmin.js
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';
import dotenv from 'dotenv';
import Admin from './model/adminSchema.js'; // ✅ correct


dotenv.config();
mongoose.connect(process.env.MONGO_URI);

const createAdmin = async () => {
  const email = "info@transpolymer.com";
  const password = "transpolymer@2025"; // Use a strong password
  const passwordHash = await bcrypt.hash(password, 10);

  try {
    const existing = await Admin.findOne({ email });
    if (existing) {
      console.log("Admin already exists");
    } else {
      const admin = new Admin({ email, passwordHash });
      await admin.save();
      console.log("✅ Admin created successfully");
    }
  } catch (err) {
    console.error(err);
  } finally {
    mongoose.disconnect();
  }
};

createAdmin();
