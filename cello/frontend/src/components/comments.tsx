import React, { useCallback, useEffect, useState } from "react";
import { errorMessage } from "../api/client";
import { Comment, deleteComment, listComments } from "../api/slidesApi";

interface CommentsProps {
  imageId?: string;
  /** bump this to force a refetch (e.g. after the viewer saves a comment) */
  refreshKey?: number;
}

const Comments: React.FC<CommentsProps> = ({ imageId, refreshKey = 0 }) => {
  const [comments, setComments] = useState<Comment[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!imageId) {
      setComments([]);
      return;
    }
    try {
      setComments(await listComments(imageId));
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, [imageId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const remove = async (id: string) => {
    try {
      await deleteComment(id);
      setComments((cs) => cs.filter((c) => c.id !== id));
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Comments</h2>
        <span className="muted">{comments.length}</span>
      </header>
      {error && <div className="notice error">{error}</div>}
      {!imageId && <p className="muted">Select a slide to see its comments.</p>}
      {imageId && comments.length === 0 && !error && <p className="muted">No comments yet. Use Annotate, draw a region, and add one.</p>}
      <ul className="comment-list">
        {comments.map((c) => (
          <li key={c.id}>
            <div className="comment-text">{c.comment}</div>
            <div className="comment-meta">
              <span>{new Date(c.createdAt).toLocaleString()}</span>
              <button className="btn btn-ghost btn-sm" onClick={() => remove(c.id)} title="DELETE /comments/:id">
                delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default Comments;
