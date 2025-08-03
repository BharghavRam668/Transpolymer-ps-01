import { useEffect, useState } from "react";
import "./AdminDashboard.css";

const API_BASE_URL = "http://localhost:3000";

function AdminDashboard() {
  const [admins, setAdmins] = useState([]);
  const [users, setUsers] = useState([]);
  const [resetRequests, setResetRequests] = useState([]);
  const [activeTab, setActiveTab] = useState("users");
  const [error, setError] = useState(null);

  // fetch data
  useEffect(() => {
    const fetchAdmins = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/admin/admin-users`, { credentials: "include" });
        if (!res.ok) throw new Error(res.statusText);
        setAdmins(await res.json());
      } catch (err) {
        console.error(err);
        setError("Failed to load admins");
      }
    };
    const fetchUsers = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/users/regular-users`, { credentials: "include" });
        if (!res.ok) throw new Error(res.statusText);
        setUsers(await res.json());
      } catch {
        setError("Failed to load users");
      }
    };
    const fetchRequests = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/reset-request/admin`, { credentials: "include" });
        if (!res.ok) throw new Error(res.statusText);
        setResetRequests(await res.json());
      } catch {
        /* no-op */
      }
    };
    fetchAdmins();
    fetchUsers();
    fetchRequests();
  }, []);

  const handleApproveReset = async id => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/approve-reset/${id}`, {
        method: "POST",
        credentials: "include",
      });
      if (res.ok) setResetRequests(r => r.filter(x => x._id !== id));
      else throw new Error(await res.text());
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDenyReset = async id => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/deny-reset/${id}`, {
        method: "POST",
        credentials: "include",
      });
      if (res.ok) setResetRequests(r => r.filter(x => x._id !== id));
      else throw new Error(await res.text());
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = async () => {
    await fetch(`${API_BASE_URL}/api/auth/logout`, { credentials: "include" });
    window.location.href = "/login";
  };

  return (
    <>
      {/* Admin Navbar */}
      <nav className="admin-navbar">
        <div className="navbar-brand">
          <h1 className="navbar-title">Admin Dashboard</h1>
          <div className="navbar-separator"></div>
          <p className="navbar-subtitle">TRANSPOLYMER</p>
        </div>
        <div className="navbar-actions">
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </nav>

      {/* Main Dashboard Content */}
      <div className="admin-dashboard">
        {error && <div className="error-message">{error}</div>}

        <div className="dashboard-tabs">
          <button className={activeTab==="users"?"tab-active":""} onClick={()=>setActiveTab("users")}>Users</button>
          <button className={activeTab==="admins"?"tab-active":""} onClick={()=>setActiveTab("admins")}>Admins</button>
          <button className={activeTab==="requests"?"tab-active":""} onClick={()=>setActiveTab("requests")}>
            Reset Requests {resetRequests.length>0 && <span className="badge">{resetRequests.length}</span>}
          </button>
        </div>

        <div className="dashboard-content">
          {activeTab==="users" && (
            <section>
              <h2>User Accounts</h2>
              {users.length===0
                ? <p>No regular users found.</p>
                : <table className="data-table">
                    <thead><tr><th>Email</th><th>Username</th></tr></thead>
                    <tbody>
                      {users.map(u=>(
                        <tr key={u._id}><td>{u.email}</td><td>{u.username||"N/A"}</td></tr>
                      ))}
                    </tbody>
                  </table>
              }
            </section>
          )}

          {activeTab==="admins" && (
            <section>
              <h2>Admin Accounts</h2>
              {admins.length===0
                ? <p>No admin users found.</p>
                : <table className="data-table">
                    <thead><tr><th>Email</th><th>Username</th><th>Role</th></tr></thead>
                    <tbody>
                      {admins.map(a=>(
                        <tr key={a._id}><td>{a.email}</td><td>{a.username||"N/A"}</td><td>{a.role||"Admin"}</td></tr>
                      ))}
                    </tbody>
                  </table>
              }
            </section>
          )}

          {activeTab==="requests" && (
            <section>
              <h2>Password Reset Requests</h2>
              {resetRequests.length===0
                ? <p>No pending requests.</p>
                : <table className="data-table">
                    <thead><tr><th>Email</th><th>Date</th><th>Actions</th></tr></thead>
                    <tbody>
                      {resetRequests.map(r=>(
                        <tr key={r._id}>
                          <td>{r.email}</td>
                          <td>{new Date(r.createdAt).toLocaleString()}</td>
                          <td className="action-buttons">
                            <button className="approve-btn" onClick={()=>handleApproveReset(r._id)}>Approve</button>
                            <button className="deny-btn" onClick={()=>handleDenyReset(r._id)}>Deny</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
              }
            </section>
          )}
        </div>
      </div>
    </>
  );
}

export default AdminDashboard;