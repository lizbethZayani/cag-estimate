class CreateChatSessions < ActiveRecord::Migration[8.0]
  def change
    create_table :chat_sessions do |t|
      t.string  :remote_session_id, null: false
      t.jsonb   :latest_metadata,   null: false, default: {}
      t.integer :turn_count,        null: false, default: 0

      t.timestamps
    end
    add_index :chat_sessions, :remote_session_id, unique: true

    add_reference :estimations, :chat_session, null: true, foreign_key: true, index: true
  end
end
