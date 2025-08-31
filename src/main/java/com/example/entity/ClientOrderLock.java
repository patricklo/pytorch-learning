package com.example.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "client_order_locks")
public class ClientOrderLock {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull(message = "Order ID is required")
    @Column(name = "order_id", nullable = false)
    private Long orderId;

    @NotBlank(message = "Locked by is required")
    @Column(name = "locked_by", nullable = false)
    private String lockedBy;

    @CreationTimestamp
    @Column(name = "locked_at", nullable = false)
    private LocalDateTime lockedAt;

    @NotNull(message = "Expires at is required")
    @Column(name = "expires_at", nullable = false)
    private LocalDateTime expiresAt;

    @NotBlank(message = "Lock token is required")
    @Column(name = "lock_token", unique = true, nullable = false)
    private String lockToken;

    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true;

    // Constructors
    public ClientOrderLock() {}

    public ClientOrderLock(Long orderId, String lockedBy, LocalDateTime expiresAt, String lockToken) {
        this.orderId = orderId;
        this.lockedBy = lockedBy;
        this.expiresAt = expiresAt;
        this.lockToken = lockToken;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getOrderId() {
        return orderId;
    }

    public void setOrderId(Long orderId) {
        this.orderId = orderId;
    }

    public String getLockedBy() {
        return lockedBy;
    }

    public void setLockedBy(String lockedBy) {
        this.lockedBy = lockedBy;
    }

    public LocalDateTime getLockedAt() {
        return lockedAt;
    }

    public void setLockedAt(LocalDateTime lockedAt) {
        this.lockedAt = lockedAt;
    }

    public LocalDateTime getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(LocalDateTime expiresAt) {
        this.expiresAt = expiresAt;
    }

    public String getLockToken() {
        return lockToken;
    }

    public void setLockToken(String lockToken) {
        this.lockToken = lockToken;
    }

    public Boolean getIsActive() {
        return isActive;
    }

    public void setIsActive(Boolean isActive) {
        this.isActive = isActive;
    }

    public boolean isExpired() {
        return LocalDateTime.now().isAfter(expiresAt);
    }

    @Override
    public String toString() {
        return "ClientOrderLock{" +
                "id=" + id +
                ", orderId=" + orderId +
                ", lockedBy='" + lockedBy + '\'' +
                ", lockedAt=" + lockedAt +
                ", expiresAt=" + expiresAt +
                ", isActive=" + isActive +
                '}';
    }
} 